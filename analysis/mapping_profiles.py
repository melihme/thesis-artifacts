from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional


DomainName = str
ProfileName = str


@dataclass(frozen=True)
class MappingProfile:
    name: str
    description: str
    wiki: Mapping[str, str]
    twitter: Mapping[str, str]

    def active_labels(self) -> list[str]:
        labels = set(self.wiki.values()) | set(self.twitter.values())
        return sorted(labels)


PROFILES: Dict[str, MappingProfile] = {
    "conservative": MappingProfile(
        name="conservative",
        description="Primary shared-label setting: PERSON, ORGANIZATION, LOCATION.",
        wiki={
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
        },
        twitter={
            "PERSON": "PERSON",
            "ORGANIZATION": "ORGANIZATION",
            "LOCATION": "LOCATION",
        },
    ),
    "extended_safe": MappingProfile(
        name="extended_safe",
        description="Conservative labels plus direct overlaps for MONEY, TIME, and PRODUCT.",
        wiki={
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "MONEY": "MONEY",
            "TIME": "TIME",
            "PRODUCT": "PRODUCT",
        },
        twitter={
            "PERSON": "PERSON",
            "ORGANIZATION": "ORGANIZATION",
            "LOCATION": "LOCATION",
            "MONEY": "MONEY",
            "TIME": "TIME",
            "PRODUCT": "PRODUCT",
        },
    ),
    "extended_risky": MappingProfile(
        name="extended_risky",
        description="Extended-safe labels plus TVSHOW, including Wiki WORK_OF_ART -> TVSHOW.",
        wiki={
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "MONEY": "MONEY",
            "TIME": "TIME",
            "PRODUCT": "PRODUCT",
            "WORK_OF_ART": "TVSHOW",
        },
        twitter={
            "PERSON": "PERSON",
            "ORGANIZATION": "ORGANIZATION",
            "LOCATION": "LOCATION",
            "MONEY": "MONEY",
            "TIME": "TIME",
            "PRODUCT": "PRODUCT",
            "TVSHOW": "TVSHOW",
        },
    ),
    "strict_direct_overlap": MappingProfile(
        name="strict_direct_overlap",
        description="Lower-bound direct-name overlap only; excludes ORG/GPE/LOC harmonization and TVSHOW.",
        wiki={
            "PERSON": "PERSON",
            "MONEY": "MONEY",
            "TIME": "TIME",
            "PRODUCT": "PRODUCT",
        },
        twitter={
            "PERSON": "PERSON",
            "MONEY": "MONEY",
            "TIME": "TIME",
            "PRODUCT": "PRODUCT",
        },
    ),
}


def require_profile(name: str) -> MappingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown mapping profile '{name}'. Available profiles: {available}") from exc


def parse_bio_label(label: str) -> tuple[Optional[str], Optional[str]]:
    if label == "O" or "-" not in label:
        return None, None
    prefix, entity_type = label.split("-", 1)
    prefix = prefix.upper()
    if prefix not in {"B", "I"}:
        return None, None
    return prefix, entity_type


def map_entity_type(entity_type: str, domain: DomainName, profile_name: ProfileName) -> Optional[str]:
    profile = require_profile(profile_name)
    if domain == "wiki":
        return profile.wiki.get(entity_type)
    if domain == "twitter":
        return profile.twitter.get(entity_type)
    raise ValueError("domain must be 'wiki' or 'twitter'")


def map_bio_label(label: str, domain: DomainName, profile_name: ProfileName) -> str:
    prefix, entity_type = parse_bio_label(label)
    if prefix is None or entity_type is None:
        return "O"
    mapped_entity = map_entity_type(entity_type, domain=domain, profile_name=profile_name)
    if mapped_entity is None:
        return "O"
    return f"{prefix}-{mapped_entity}"


def map_bio_sequence(labels: Iterable[str], domain: DomainName, profile_name: ProfileName) -> list[str]:
    return [map_bio_label(label, domain=domain, profile_name=profile_name) for label in labels]


def profile_summary() -> dict[str, dict[str, object]]:
    return {
        name: {
            "description": profile.description,
            "active_labels": profile.active_labels(),
            "wiki": dict(profile.wiki),
            "twitter": dict(profile.twitter),
        }
        for name, profile in sorted(PROFILES.items())
    }
