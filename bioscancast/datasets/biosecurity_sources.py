"""Known biosecurity dashboard URLs by pathogen.

v1 — flagged for iteration after first benchmark run.
The dashboard list and routing logic will need updating as new outbreaks emerge
and data portals change.
"""

DASHBOARD_LOOKUP: dict[str, list[str]] = {
    "h5n1": [
        "https://www.cdc.gov/bird-flu/situation-summary/",
        "https://www.who.int/teams/global-influenza-programme/avian-influenza",
    ],
    "avian influenza": [
        "https://www.cdc.gov/bird-flu/situation-summary/",
        "https://www.who.int/teams/global-influenza-programme/avian-influenza",
    ],
    "mpox": [
        "https://ourworldindata.org/mpox",
        "https://www.who.int/emergencies/situation-reports",
        "https://www.cdc.gov/mpox/data-research/index.html",
    ],
    "ebola": [
        "https://www.afro.who.int/health-topics/ebola-virus-disease",
        "https://www.cdc.gov/ebola/index.html",
    ],
    "covid-19": [
        "https://ourworldindata.org/coronavirus",
        "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports",
    ],
    "marburg": [
        "https://www.who.int/news-room/fact-sheets/detail/marburg-virus-disease",
        "https://www.cdc.gov/marburg/index.html",
    ],
}
