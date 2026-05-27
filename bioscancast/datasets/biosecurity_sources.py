"""Known biosecurity dashboard URLs by pathogen.

v1 — flagged for iteration after first benchmark run. The dashboard list
and routing logic will need updating as new outbreaks emerge and data
portals change.

Each entry carries a pathogen-specific ``title`` and ``snippet`` so that
the heuristic filter and the LLM-rescue path have real signal to work
with. The earlier convention ("Dashboard: cdc.gov" with a generic
snippet) produced keyword_overlap_score = 0.000 across the board — see
issue #14 and the q7/q12 live-run findings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardEntry:
    """A curated authoritative source for a pathogen.

    The title and snippet are intended to be readable as a search result
    in their own right: pathogen name, the kind of data the page hosts,
    and the publisher. They feed both the keyword-overlap heuristic and
    the LLM-rescue path.
    """

    url: str
    title: str
    snippet: str


DASHBOARD_LOOKUP: dict[str, list[DashboardEntry]] = {
    "h5n1": [
        DashboardEntry(
            url="https://www.cdc.gov/bird-flu/situation-summary/",
            title="CDC H5N1 bird flu situation summary: human cases and outbreaks in the United States",
            snippet="CDC tracking of H5N1 avian influenza human cases, affected livestock herds, and public-health response in the US.",
        ),
        DashboardEntry(
            url="https://www.who.int/teams/global-influenza-programme/avian-influenza",
            title="WHO Global Influenza Programme: avian influenza A(H5N1) human cases and surveillance",
            snippet="WHO monitoring of human H5N1 cases, animal-to-human spillover events, and global surveillance reporting.",
        ),
    ],
    "avian influenza": [
        DashboardEntry(
            url="https://www.cdc.gov/bird-flu/situation-summary/",
            title="CDC H5N1 bird flu situation summary: human cases and outbreaks in the United States",
            snippet="CDC tracking of H5N1 avian influenza human cases, affected livestock herds, and public-health response in the US.",
        ),
        DashboardEntry(
            url="https://www.who.int/teams/global-influenza-programme/avian-influenza",
            title="WHO Global Influenza Programme: avian influenza A(H5N1) human cases and surveillance",
            snippet="WHO monitoring of human H5N1 cases, animal-to-human spillover events, and global surveillance reporting.",
        ),
    ],
    "mpox": [
        DashboardEntry(
            url="https://ourworldindata.org/mpox",
            title="Our World in Data mpox tracker: global confirmed cases and deaths",
            snippet="OWID dashboard tracking cumulative confirmed mpox cases and deaths globally, broken down by country and region, updated from national health agencies.",
        ),
        DashboardEntry(
            url="https://www.who.int/emergencies/situation-reports",
            title="WHO situation reports including the multi-country mpox outbreak",
            snippet="WHO situation reports with weekly case counts, country breakdowns, and public-health guidance for ongoing outbreaks including mpox.",
        ),
        DashboardEntry(
            url="https://www.cdc.gov/mpox/data-research/index.html",
            title="CDC mpox data and research dashboard for the United States",
            snippet="CDC tracking of US mpox cases, demographic data, vaccination coverage, and outbreak response.",
        ),
    ],
    "ebola": [
        DashboardEntry(
            url="https://www.afro.who.int/health-topics/ebola-virus-disease",
            title="WHO Africa Ebola virus disease outbreak surveillance and case counts",
            snippet="WHO regional office for Africa tracking of Ebola virus disease outbreaks, confirmed and suspected cases, deaths, and response across African countries.",
        ),
        DashboardEntry(
            url="https://www.cdc.gov/ebola/index.html",
            title="CDC Ebola virus disease outbreak history and case counts",
            snippet="CDC information on current and historical Ebola virus disease outbreaks worldwide, with case counts, deaths, and US public-health response.",
        ),
    ],
    "covid-19": [
        DashboardEntry(
            url="https://ourworldindata.org/coronavirus",
            title="Our World in Data COVID-19 tracker: global cases, deaths, and vaccinations",
            snippet="OWID dashboard tracking cumulative COVID-19 confirmed cases, deaths, hospitalizations, and vaccination coverage globally by country.",
        ),
        DashboardEntry(
            url="https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports",
            title="WHO COVID-19 situation reports and global case counts",
            snippet="WHO situation reports with updates on COVID-19 confirmed cases, deaths, variant tracking, and country-level data.",
        ),
    ],
    "marburg": [
        DashboardEntry(
            url="https://www.who.int/news-room/fact-sheets/detail/marburg-virus-disease",
            title="WHO Marburg virus disease facts and outbreak case counts",
            snippet="WHO factsheet on Marburg virus disease including transmission, symptoms, case-fatality ratio, and historical outbreak case and death counts.",
        ),
        DashboardEntry(
            url="https://www.cdc.gov/marburg/index.html",
            title="CDC Marburg virus disease outbreaks and surveillance",
            snippet="CDC information on Marburg virus disease outbreaks worldwide, case counts, deaths, and US public-health surveillance.",
        ),
    ],
}
