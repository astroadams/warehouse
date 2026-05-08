from __future__ import annotations

import time
from datetime import date
from typing import Any, Iterable

import requests
from shapely.geometry import Polygon, box
from tqdm import tqdm

from warehouse_growth.data_sources import VectorFeature

# overpass-api.de has been blocking cloud-provider IP ranges since April 2026.
# private.coffee is a reliable no-rate-limit alternative.
# Other options: https://overpass.osm.ch/api/interpreter (Swiss instance)
_OVERPASS_URL = "https://overpass.private.coffee/api/interpreter"
_DEFAULT_TIMEOUT = 180

# Overpass enforces stricter header checks to deter automated abuse.
# A descriptive User-Agent and explicit Accept avoid 406 rejections.
_REQUEST_HEADERS = {
    "User-Agent": "warehouse-growth/0.1 (geospatial research)",
    "Accept": "application/json",
}

# Only `way` elements are handled here. Multipolygon `relation` buildings are rare
# for large industrial/warehouse structures, so this covers the practical majority.
# Extend to relations if completeness in dense urban areas matters.
_QUERY_TEMPLATE = """\
[out:json][timeout:{timeout}]{date_setting};
(
  way["building"]({bbox});
);
out body geom;
"""

_ROAD_QUERY_TEMPLATE = """\
[out:json][timeout:{timeout}];
(
{ways}
);
out body geom;
"""


class OverpassRoadSource:
    """OSM road and rail features via the Overpass API.

    Queries for highway and railway ways within the AOI bounding box.
    Yields a VectorFeature per element with its Shapely geometry and raw OSM tags.
    Pass the results to ``build_road_mask`` / ``aoi_from_road_mask`` in road_mask.py.
    """

    def __init__(self, url: str = _OVERPASS_URL, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.url = url
        self.timeout = timeout

    def roads_for_aoi(self, aoi: Any, classes: list[str]) -> Iterable[VectorFeature]:
        """Yield road/rail features intersecting the AOI.

        *classes* contains OSM highway values (e.g. ``"motorway"``, ``"trunk"``)
        and/or the special strings ``"rail"`` (railway ways) and ``"port"``
        (landuse=port ways and ferry_terminal nodes).
        """
        from shapely.geometry import LineString, Point

        minx, miny, maxx, maxy = aoi.bounds
        bbox_str = f"{miny},{minx},{maxy},{maxx}"

        highway_classes = [c for c in classes if c not in ("rail", "port")]
        way_lines: list[str] = []
        if highway_classes:
            pattern = "|".join(highway_classes)
            way_lines.append(f'  way["highway"~"{pattern}"]({bbox_str});')
        if "rail" in classes:
            way_lines.append(f'  way["railway"~"rail|light_rail|subway|tram|monorail"]({bbox_str});')
        if "port" in classes:
            way_lines.append(f'  way["landuse"="port"]({bbox_str});')
            way_lines.append(f'  node["amenity"="ferry_terminal"]({bbox_str});')

        if not way_lines:
            return

        query = _ROAD_QUERY_TEMPLATE.format(timeout=self.timeout, ways="\n".join(way_lines))

        _transient = {429, 503, 504}
        response = None
        for attempt in range(4):
            if attempt:
                wait = 2 ** attempt
                time.sleep(wait)
            try:
                response = requests.post(
                    self.url, data={"data": query}, headers=_REQUEST_HEADERS, timeout=self.timeout
                )
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                if attempt == 3:
                    raise
                tqdm.write(f"  {type(exc).__name__} on attempt {attempt + 1}, will retry")
                continue
            if response.status_code not in _transient:
                break
        response.raise_for_status()

        seen_ids: set[int] = set()
        for el in response.json().get("elements", []):
            el_id = el.get("id")
            if el_id in seen_ids:
                continue
            seen_ids.add(el_id)

            el_type = el.get("type")
            if el_type == "way":
                coords = [(node["lon"], node["lat"]) for node in el.get("geometry", [])]
                if len(coords) < 2:
                    continue
                geom: Any = LineString(coords)
            elif el_type == "node":
                geom = Point(el["lon"], el["lat"])
            else:
                continue

            if not geom.intersects(aoi):
                continue
            yield VectorFeature(geometry=geom, properties=el.get("tags", {}))


class OverpassTagSource:
    """OSM building tags via the Overpass API.

    Queries for all `building=*` ways within the AOI bounding box and yields
    a VectorFeature per way with its shapely Polygon geometry and raw OSM tags.
    Pass the results to `label_from_osm_tags` or `label_footprints` in labels.py.
    """

    def __init__(self, url: str = _OVERPASS_URL, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.url = url
        self.timeout = timeout

    def tags_for_aoi(
        self,
        aoi: Any,
        *,
        grid: int = 4,
        as_of_date: date | None = None,
    ) -> Iterable[VectorFeature]:
        """Yield OSM building features intersecting the AOI.

        `grid` splits the AOI into a grid×grid mesh of sub-queries (default 2×2=4
        tiles). Increase it if you get 406 errors on a large AOI; each tile must fit
        within Overpass's single-query area limit (~0.25 deg² works reliably).

        `as_of_date` returns the historical state of OSM on that date via the
        Overpass [date:...] setting.  When None, the current live state is returned.
        """
        date_setting = f'[date:"{as_of_date}T23:59:59Z"]' if as_of_date else ""

        minx, miny, maxx, maxy = aoi.bounds
        x_step = (maxx - minx) / grid
        y_step = (maxy - miny) / grid
        tiles = [
            box(minx + j * x_step, miny + i * y_step,
                minx + (j + 1) * x_step, miny + (i + 1) * y_step)
            for i in range(grid)
            for j in range(grid)
        ]

        seen_ids: set[int] = set()
        _transient = {429, 503, 504}

        for tile in tqdm(tiles, desc="Querying Overpass", unit=" tile", leave=True):
            tmx, tmy, tmxx, tmyy = tile.bounds
            # Overpass bbox order: south, west, north, east
            bbox = f"{tmy},{tmx},{tmyy},{tmxx}"
            query = _QUERY_TEMPLATE.format(timeout=self.timeout, bbox=bbox, date_setting=date_setting)

            response = None
            for attempt in range(4):
                if attempt:
                    wait = 2 ** attempt
                    tqdm.write(f"  retry {attempt} after {wait}s …")
                    time.sleep(wait)
                try:
                    response = requests.post(
                        self.url, data={"data": query}, headers=_REQUEST_HEADERS, timeout=self.timeout
                    )
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                    if attempt == 3:
                        raise
                    tqdm.write(f"  {type(exc).__name__} on attempt {attempt + 1}, will retry")
                    continue
                if response.status_code not in _transient:
                    break
            response.raise_for_status()

            for el in response.json().get("elements", []):
                if el.get("type") != "way":
                    continue
                el_id = el.get("id")
                if el_id in seen_ids:
                    continue
                seen_ids.add(el_id)

                coords = [(node["lon"], node["lat"]) for node in el.get("geometry", [])]
                if len(coords) < 4:
                    continue
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if not geom.intersects(aoi):
                    continue
                yield VectorFeature(geometry=geom, properties=el.get("tags", {}))
