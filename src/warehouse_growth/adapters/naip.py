from __future__ import annotations

import time
from typing import Any, Iterable

import planetary_computer as pc
from tqdm import tqdm

from warehouse_growth.data_sources import ImageryAsset

# Microsoft Planetary Computer hosts NAIP as COGs on Azure Blob Storage.
# Asset URLs must be signed with a free SAS token via the `planetary-computer`
# package; no AWS credentials or requester-pays charges involved.
_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_COLLECTION = "naip"
_RESOLUTION_M = 1.0
_PAGE_SIZE = 100
_TIMEOUT = 90
_MAX_RETRIES = 6


def _parse_crs(properties: dict) -> str | None:
    """Extract a CRS string from STAC item properties."""
    code = properties.get("proj:code") or properties.get("proj:epsg")
    if not code:
        return None
    if isinstance(code, int):
        return f"EPSG:{code}"
    s = str(code).strip()
    return s if s.upper().startswith("EPSG:") else f"EPSG:{s}"


def _sign_href(href: str) -> str:
    """Return a SAS-token-signed URL for Microsoft Planetary Computer assets."""
    return pc.sign(href)


def _post_with_retry(url: str, payload: dict) -> dict:
    """POST to a STAC endpoint, retrying on 429/503/504 with backoff."""
    import requests

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        except requests.exceptions.Timeout:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        if resp.status_code in (429, 503, 504):
            if attempt == _MAX_RETRIES - 1:
                resp.raise_for_status()
            time.sleep(2 ** attempt)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("Unreachable")


class NAIPImagerySource:
    """NAIP imagery assets via the Microsoft Planetary Computer STAC catalog.

    Each returned ImageryAsset points to a signed Azure Blob COG href for the
    four-band (RGB+NIR) GeoTIFF. Rasterio can stream these directly over HTTPS.

    Install the signing dependency with:
        pip install planetary-computer
    """

    def __init__(self, stac_url: str = _STAC_SEARCH_URL) -> None:
        self.stac_url = stac_url

    def assets_for_aoi(self, aoi: Any, epoch: str) -> Iterable[ImageryAsset]:
        # Serialize AOI geometry to GeoJSON for the POST body.
        try:
            geojson_geom = aoi.__geo_interface__
        except AttributeError:
            import shapely.geometry
            geojson_geom = shapely.geometry.mapping(aoi)

        base_payload = {
            "collections": [_COLLECTION],
            "intersects": geojson_geom,
            "datetime": f"{epoch}-01-01T00:00:00Z/{epoch}-12-31T23:59:59Z",
            "limit": _PAGE_SIZE,
        }

        # Collect all pages up front so tqdm can show a total.
        all_items: list[dict] = []
        token = None
        while True:
            payload = {**base_payload}
            if token:
                payload["token"] = token

            page = _post_with_retry(self.stac_url, payload)
            features = page.get("features", [])
            all_items.extend(features)

            # MPC uses a "next" link or a continuation token for pagination.
            next_link = next(
                (lnk for lnk in page.get("links", []) if lnk.get("rel") == "next"),
                None,
            )
            if not next_link or not features:
                break
            # Extract token from the next link's body or href.
            token = (next_link.get("body") or {}).get("token")
            if not token:
                # Fall back to parsing the token query param from the href.
                href = next_link.get("href", "")
                for part in href.split("&"):
                    if part.startswith("token="):
                        token = part[6:]
                        break
            if not token:
                break  # no token found — cannot paginate further

        for feature in tqdm(all_items, desc=f"NAIP tiles ({epoch})", unit=" tile", leave=True):
            props = feature.get("properties", {})
            assets = feature.get("assets", {})
            asset = assets.get("image")
            if asset is None:
                continue

            href = _sign_href(asset["href"])
            crs = _parse_crs(props)
            capture = props.get("datetime", "")[:10] or None

            yield ImageryAsset(
                uri=href,
                epoch=epoch,
                capture_date=capture,
                crs=crs,
                resolution_meters=_RESOLUTION_M,
            )
