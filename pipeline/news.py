"""Headline preparation: cleaning, issuer filter, roundup filter and FinBERT scores."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from email.utils import parsedate_to_datetime

from .config import DATA, PARAMS

BUSINESS = (r"share|stock|investor|profit|earning|revenue|sales|quarter|dividend|buyback|guidance|outlook|forecast|"
            r"ceo|chief executive|cfo|chairman|board|merger|acqui|takeover|deal|bid|stake|spin.?off|ipo|bond|debt|"
            r"rating|downgrade|upgrade|analyst|price target|billion|million|\$|€|£|market|company|firm|corp|"
            r"business|lawsuit|sue|court|settle|fine|probe|regulator|antitrust|layoff|job cut|strike|union|plant|"
            r"factory|launch|unveil|product|customer|contract|order|supplier|recall|expan|invest|results|compan|split|"
            r"unit|division|sell|buy|acquir|merg|cut|hire|cash|growth|loss|gain|rise|rises|fall|jump|slump|drop|"
            r"surge|soar|plunge|tumble|rally|beat|miss|record|q[1-4]\b")
SECTOR = {
    "Information Technology": r"software|cloud|chip|semiconductor|smartphone|iphone|ipad|mac|app|device|data|ai\b|"
                              r"artificial intelligence|tech|computer|server|network|payment|card",
    "Energy": r"oil|gas|lng|energy|petroleum|refin|drill|offshore|fuel|crude|barrel|pipeline|shale|well|solar|"
              r"renewable|power|electric|hydrogen|carbon|opec",
    "Health Care": r"drug|pharma|medic|biotech|therap|trial|patient|fda|ema|vaccine|health|cancer|clinical|device|"
                   r"treatment|approval",
    "Financials": r"bank|lender|loan|mortgage|deposit|insur|asset|capital|credit|fund|trading|wealth",
    "Consumer Discretionary": r"store|retail|shopper|car|vehicle|auto|ev\b|electric|brand|fashion|apparel|sneaker|"
                              r"restaurant|menu|park|streaming|movie|film|box office|luxury|e.?commerce|delivery",
    "Consumer Staples": r"store|retail|shopper|grocery|food|beverage|drink|beer|brand|consumer|price|tobacco|"
                        r"cigarette|pharmacy",
    "Industrials": r"aircraft|plane|jet|airline|defen[cs]e|pentagon|engine|rail|freight|shipping|parcel|logistic|"
                   r"machin|equipment|construction|industrial|electric|power|automation",
    "Materials": r"chemical|seed|crop|fertili|plastic|mining|copper|gold|steel|gas|plant",
    "Telecommunication Services": r"telecom|wireless|mobile|network|5g|broadband|fiber|fibre|carrier|subscriber|"
                                  r"spectrum|phone",
    "Utilities": r"power|electric|utility|grid|nuclear|renewable|wind|solar|gas|energy|customer",
}
# Phrases that name something else than the issuer.
EXCLUDE = {
    "AAPL": r"apple (?:pie|tree|cider|juice|orchard|picking|harvest|variet|crisp)|big apple|fiona apple|orchard",
    "AMZN": r"rainforest|amazon (?:river|basin|fire|forest|region)|amazonia|deforest",
    "ABT": r"abbott elementary|greg abbott|tony abbott|governor abbott|abbott and costello",
    "BAX": r"baxter black|baxter dury",
    "CAT": r"caterpillar (?:larva|insect|moth|butterfl)|hungry caterpillar",
    "CVX": r"chevron (?:pattern|print|nail|stitch|deference|doctrine)|crochet|knit",
    "CL": r"colgate (?:university|raiders)",
    "C": r"citi (?:field|open|bike)",
    "DVN": r"",
    "EMR": r"emerson college|ralph waldo",
    "F": r"(?:harrison|gerald|tom|doug|rob|betty|lauren|henry ford hospital|whitey|john) ford|ford (?:foundation|field|theatre|theater)",
    "GM": r"gm (?:crop|food|mosquito|salmon|seed|wheat)|genetically",
    "HPQ": r"\d+\s?hp\b|horsepower|hewlett packard enterprise|\bhpe\b",
    "LLY": r"lilly (?:allen|james|collins|singh|reinhart)|evangeline lilly",
    "LOW": r"rob lowe|chris lowe|zane lowe",
    "META": r"meta.?analysis|metaverse fashion|metamorph",
    "MRK": r"merck kgaa|merck group|darmstadt|\bemd\b",
    "NKE": r"goddess",
    "ORCL": r"oracle of omaha|oracle bone",
    "OXY": r"occidental college",
    "TGT": r"price target|target price|on target|targets|targeted|targeting|shooting target|inflation target|target date",
    "V": r"visas|visa (?:application|waiver|free|requirement|holder|ban|rule|program|interview|appointment|fee|policy|"
         r"applicant|overstay|denial|denied|process)|student visa|work visa|h-1b|golden visa|e-visa|immigra|travel visa|"
         r"tourist visa|asylum",
    "ALV.DE": r"allianz (?:arena|stadium|field|park|parque|riviera|cup)",
    "BAYN.DE": r"leverkusen|bundesliga",
    "BMW.DE": r"bmw (?:championship|pga|open|international open)",
    "MBG.DE": r"formula (?:one|1)|\bf1\b|grand prix|hamilton|wolff|mercedes amg petronas|mercedes-amg f1|mercedes team",
    "NOKIA.HE": r"\bhmd\b|nokia \d|nokia (?:lumia )?phone",
    "SAN.MC": r"racing santander|santander (?:cathedral|beach)|weather santander",
    "SAP.DE": r"tree sap|maple sap|sap.?suck|sap feeding",
    "DG.PA": r"da vinci|vinci code|leonardo",
    "ORA.PA": r"orange (?:county|juice|is the new black|bowl|line)|agent orange|orange alert|code orange",
    "PHIA.AS": r"phillips",
    "TTE.PA": r"total (?:war|eclipse|recall|solar)",
}
TRAILING_SOURCE = re.compile(r"\s+[-–|]\s+[^-–|]{2,80}$")


def clean_title(title, source=None):
    """Normalise a Google News title and remove the trailing ' - Source' tag."""
    text = unicodedata.normalize("NFKC", title or "").replace("’", "'").replace("‘", "'").strip()
    if source and text.endswith(f" - {source}"):
        return text[: -len(source) - 3].strip()
    return TRAILING_SOURCE.sub("", text).strip()


def score_key(title):
    return hashlib.sha256(f"finbert|{PARAMS['finbert_model']}|{title}".encode()).hexdigest()


def _parse_alias(alias):
    """Alias markers: '=' case-sensitive, '?' needs business or industry context, '!' needs industry context.
    Short or all-caps aliases (GE, VW, ING) are always case-sensitive."""
    need, case = None, False
    while alias[:1] in ("=", "?", "!"):
        marker, alias = alias[0], alias[1:]
        if marker == "?":
            need = "any"
        elif marker == "!":
            need = "industry"
        else:
            case = True
    case = case or alias.isupper() or len(alias) <= 3 or any(c.isdigit() for c in alias)
    pattern = re.compile(rf"(?<![\w&]){re.escape(alias)}(?![\w&])", 0 if case else re.I)
    return pattern, need


class IssuerFilter:
    """Accept a headline only if it names the issuer; names that are also common words or
    names of people and places additionally require business or industry context."""

    def __init__(self, universe):
        self.rules, self._all = {}, []
        for firm in universe:
            ticker = firm["ticker"]
            aliases = [_parse_alias(a.strip()) for a in firm["aliases"].split(";") if a.strip()]
            industry = SECTOR.get(firm["sector"], "")
            self.rules[ticker] = {
                "aliases": aliases,
                "any": re.compile(rf"(?:{BUSINESS}|{industry})", re.I),
                "industry": re.compile(rf"(?:{industry})", re.I),
                "exclude": re.compile(EXCLUDE[ticker], re.I) if EXCLUDE.get(ticker) else None,
            }
            self._all += [(ticker, pattern) for pattern, need in aliases if need is None]

    def accept(self, ticker, text):
        rule = self.rules[ticker]
        if not text or (rule["exclude"] is not None and rule["exclude"].search(text)):
            return False
        for pattern, need in rule["aliases"]:
            if pattern.search(text) and (need is None or rule[need].search(text)):
                return True
        return False

    def firms_named(self, text):
        """Firms named unambiguously in a headline (used to detect market roundups)."""
        return {t for t, pattern in self._all if pattern.search(text)}


def load_headlines(universe, audit):
    """Read the stored Google News windows, clean, de-duplicate and filter headlines."""
    issuer = IssuerFilter(universe)
    counts, records, seen = Counter(), [], set()
    for firm in universe:
        path = DATA / "headlines" / f"{firm['ticker']}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            window = json.loads(line)
            counts["query_windows"] += 1
            counts["capped_windows"] += window["n"] >= 100
            for item in window["items"]:
                counts["retrieved"] += 1
                try:
                    stamp = parsedate_to_datetime(item["pub"])
                except (TypeError, ValueError):
                    continue
                day = stamp.date().isoformat()
                title = clean_title(item["title"], item.get("src"))
                key = (firm["ticker"], title.casefold(), day)
                if not title or key in seen:
                    continue
                seen.add(key)
                counts["unique"] += 1
                if not PARAMS["news_start"] <= day <= PARAMS["observation_through_date"]:
                    continue
                counts["in_range"] += 1
                if not issuer.accept(firm["ticker"], title):
                    continue
                counts["issuer"] += 1
                if len(issuer.firms_named(title)) >= PARAMS["roundup_min_firms"]:
                    counts["roundup"] += 1
                    continue
                records.append({"ticker": firm["ticker"], "date": day, "title": title})
    audit.update({f"headlines_{k}": v for k, v in counts.items()})
    audit["headlines_accepted"] = len(records)
    return records


def attach_scores(records, audit):
    """Join FinBERT probabilities (data/finbert_scores.jsonl); tone s = p+ - p-."""
    scores = {}
    path = DATA / "finbert_scores.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        scores[row["key"]] = row
    out, missing = [], 0
    for record in records:
        row = scores.get(score_key(record["title"]))
        if row is None:
            missing += 1
            continue
        out.append({**record, "positive": row["positive"], "negative": row["negative"],
                    "tone": row["positive"] - row["negative"]})
    audit["headlines_unscored"] = missing
    if missing:
        raise ValueError(f"{missing} accepted headlines have no FinBERT score; run data_collection/score_finbert.py")
    return out
