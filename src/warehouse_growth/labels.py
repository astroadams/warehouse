from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from warehouse_growth.data_sources import VectorFeature


class BuildingLabel(str, Enum):
    WAREHOUSE = "warehouse"
    NON_WAREHOUSE = "non_warehouse"
    AMBIGUOUS_INDUSTRIAL = "ambiguous_industrial"


@dataclass(frozen=True)
class BuildingInstance:
    geometry: Any
    label: BuildingLabel
    source_id: str | None = None
    epoch: str | None = None


# OSM `building=` values that map to each label class.
_WAREHOUSE_TAGS = frozenset({"warehouse", "logistics", "distribution_center", "storage"})
_INDUSTRIAL_TAGS = frozenset({"industrial", "manufacture", "factory", "works", "shed"})


def label_from_osm_tags(tags: dict) -> BuildingLabel:
    """Map OSM building tags to a BuildingLabel.

    Unrecognised or absent tags return NON_WAREHOUSE rather than AMBIGUOUS so
    that the ambiguous bucket stays reserved for known-industrial-but-unclear cases.
    """
    building = tags.get("building", "").lower().strip()
    if building in _WAREHOUSE_TAGS:
        return BuildingLabel.WAREHOUSE
    if building in _INDUSTRIAL_TAGS:
        return BuildingLabel.AMBIGUOUS_INDUSTRIAL
    return BuildingLabel.NON_WAREHOUSE


def label_footprints(
    footprints: Iterable[VectorFeature],
    tags: Iterable[VectorFeature],
    epoch: str | None = None,
) -> list[BuildingInstance]:
    """Assign labels to Microsoft footprints by spatial join with OSM tag features.

    Each footprint is matched to the OSM feature with the greatest overlap area.
    Footprints with no OSM match are labelled NON_WAREHOUSE.

    Uses shapely 2.0's vectorised bulk STRtree query so the spatial index is
    traversed once for all footprints rather than once per footprint.
    """
    try:
        from collections import defaultdict

        from shapely.strtree import STRtree
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError("Install geo extras: pip install warehouse-growth[geo]") from e

    tag_list = list(tags)
    fp_list = list(footprints)

    if not tag_list:
        return [
            BuildingInstance(geometry=fp.geometry, label=BuildingLabel.NON_WAREHOUSE, epoch=epoch)
            for fp in fp_list
        ]

    tag_geoms = [t.geometry for t in tag_list]
    tree = STRtree(tag_geoms)

    # Single vectorised call returns (fp_indices, tag_indices) for every
    # intersecting pair — equivalent to a spatial join.
    fp_geoms = [fp.geometry for fp in fp_list]
    fp_idxs, tag_idxs = tree.query(fp_geoms, predicate="intersects")

    matches: dict[int, list[int]] = defaultdict(list)
    for fp_i, tag_i in zip(fp_idxs.tolist(), tag_idxs.tolist()):
        matches[fp_i].append(tag_i)

    instances: list[BuildingInstance] = []
    for i, fp in enumerate(tqdm(fp_list, desc="Labelling footprints", unit=" fp", leave=True)):
        candidates = matches.get(i)
        if not candidates:
            label = BuildingLabel.NON_WAREHOUSE
        elif len(candidates) == 1:
            label = label_from_osm_tags(tag_list[candidates[0]].properties)
        else:
            best = max(candidates, key=lambda j: fp.geometry.intersection(tag_geoms[j]).area)
            label = label_from_osm_tags(tag_list[best].properties)
        instances.append(BuildingInstance(geometry=fp.geometry, label=label, epoch=epoch))
    return instances


def filter_trainable_labels(instances: list[BuildingInstance]) -> list[BuildingInstance]:
    """Drop ambiguous instances from binary warehouse training sets."""
    return [item for item in instances if item.label is not BuildingLabel.AMBIGUOUS_INDUSTRIAL]
