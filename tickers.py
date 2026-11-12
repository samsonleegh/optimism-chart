"""Stock universe (SG = SGX `.SI`, US = Nasdaq/NYSE) + sector map.

yfinance is the data source, so this is the practical investable set: liquid
counters with a clean multi-year history. Each name is tagged with a *sub-sector*
(see SECTORS) so the Sector Rotation tab can show fine-grained rotation; the
universe is deliberately broad so each sub-sector holds enough names to be
meaningful. Add `(ticker, name)` pairs (and a SECTORS entry) to widen coverage.
"""

SG_STOCKS = [
    ("^STI", "STI Index"),                 # the Straits Times Index itself (benchmark)
    # banks / financials
    ("D05.SI", "DBS Group"),
    ("O39.SI", "OCBC"),
    ("U11.SI", "UOB"),
    ("S68.SI", "SGX"),
    ("G07.SI", "Great Eastern"),
    ("AIY.SI", "iFAST"),
    # telecom
    ("Z74.SI", "Singtel"),
    ("CC3.SI", "StarHub"),
    ("CJLU.SI", "NetLink NBS Trust"),
    # transport
    ("C6L.SI", "Singapore Airlines"),
    ("S58.SI", "SATS"),
    ("S59.SI", "SIA Engineering"),
    ("C52.SI", "ComfortDelGro"),
    ("S61.SI", "SBS Transit"),
    # property developers
    ("9CI.SI", "CapitaLand Investment"),
    ("C09.SI", "City Developments"),
    ("U14.SI", "UOL Group"),
    ("F17.SI", "GuocoLand"),
    ("U06.SI", "Singapore Land Group"),
    ("H78.SI", "Hongkong Land"),
    ("TQ5.SI", "Frasers Property"),
    ("Z25.SI", "Yanlord Land"),
    # REITs (by type)
    ("A17U.SI", "CapitaLand Ascendas REIT"),
    ("M44U.SI", "Mapletree Logistics Trust"),
    ("ME8U.SI", "Mapletree Industrial Trust"),
    ("BUOU.SI", "Frasers Logistics & Commercial Trust"),
    ("J91U.SI", "ESR-LOGOS REIT"),
    ("O5RU.SI", "AIMS APAC REIT"),
    ("C38U.SI", "CapitaLand Integrated Commercial Trust"),
    ("CRPU.SI", "Sasseur REIT"),
    ("J69U.SI", "Frasers Centrepoint Trust"),
    ("AU8U.SI", "CapitaLand China Trust"),
    ("SK6U.SI", "Paragon REIT"),
    ("JYEU.SI", "Lendlease Global REIT"),
    ("N2IU.SI", "Mapletree Pan Asia Commercial Trust"),
    ("K71U.SI", "Keppel REIT"),
    ("T82U.SI", "Suntec REIT"),
    ("AJBU.SI", "Keppel DC REIT"),
    ("DCRU.SI", "Digital Core REIT"),
    ("HMN.SI", "CapitaLand Ascott Trust"),
    ("Q5T.SI", "Far East Hospitality Trust"),
    ("J85.SI", "CDL Hospitality Trusts"),
    ("C2PU.SI", "ParkwayLife REIT"),
    # industrials
    ("BN4.SI", "Keppel"),
    ("J36.SI", "Jardine Matheson"),
    ("S63.SI", "ST Engineering"),
    ("BS6.SI", "Yangzijiang Shipbuilding"),
    ("5E2.SI", "Seatrium"),
    ("S08.SI", "Singapore Post"),
    ("U96.SI", "Sembcorp Industries"),
    # agriculture
    ("F34.SI", "Wilmar International"),
    ("VC2.SI", "Olam Group"),
    ("584.SI", "First Resources"),
    ("P8Z.SI", "Bumitama Agri"),
    ("E5H.SI", "Golden Agri-Resources"),
    # consumer
    ("D01.SI", "DFI Retail Group"),
    ("Y92.SI", "Thai Beverage"),
    ("OV8.SI", "Sheng Siong Group"),
    ("C07.SI", "Jardine Cycle & Carriage"),
    ("G13.SI", "Genting Singapore"),
    # technology
    ("V03.SI", "Venture Corp"),
    ("AWX.SI", "AEM Holdings"),
    # healthcare
    ("Q0F.SI", "IHH Healthcare"),
    ("BVA.SI", "Top Glove"),
    ("BSL.SI", "Raffles Medical"),
]

US_STOCKS = [
    # software
    ("MSFT", "Microsoft"),
    ("ORCL", "Oracle"),
    ("CRM", "Salesforce"),
    ("ADBE", "Adobe"),
    ("NOW", "ServiceNow"),
    ("INTU", "Intuit"),
    # semiconductors
    ("NVDA", "NVIDIA"),
    ("AVGO", "Broadcom"),
    ("AMD", "Advanced Micro Devices"),
    ("INTC", "Intel"),
    ("QCOM", "Qualcomm"),
    ("TXN", "Texas Instruments"),
    ("MU", "Micron Technology"),
    # internet & media
    ("GOOGL", "Alphabet"),
    ("META", "Meta Platforms"),
    ("NFLX", "Netflix"),
    ("DIS", "Walt Disney"),
    ("CMCSA", "Comcast"),
    # tech hardware
    ("AAPL", "Apple"),
    ("DELL", "Dell Technologies"),
    # banks
    ("JPM", "JPMorgan Chase"),
    ("BAC", "Bank of America"),
    ("WFC", "Wells Fargo"),
    ("C", "Citigroup"),
    ("GS", "Goldman Sachs"),
    ("MS", "Morgan Stanley"),
    # payments & financial services
    ("V", "Visa"),
    ("MA", "Mastercard"),
    ("AXP", "American Express"),
    ("PYPL", "PayPal"),
    # insurance
    ("BRK-B", "Berkshire Hathaway"),
    ("PGR", "Progressive"),
    ("CB", "Chubb"),
    # consumer staples
    ("WMT", "Walmart"),
    ("PG", "Procter & Gamble"),
    ("COST", "Costco"),
    ("KO", "Coca-Cola"),
    ("PEP", "PepsiCo"),
    ("MDLZ", "Mondelez"),
    ("CL", "Colgate-Palmolive"),
    ("PM", "Philip Morris"),
    # consumer discretionary
    ("AMZN", "Amazon"),
    ("HD", "Home Depot"),
    ("LOW", "Lowe's"),
    ("NKE", "Nike"),
    ("TGT", "Target"),
    # restaurants
    ("MCD", "McDonald's"),
    ("SBUX", "Starbucks"),
    ("CMG", "Chipotle"),
    # autos
    ("TSLA", "Tesla"),
    ("GM", "General Motors"),
    ("F", "Ford"),
    # healthcare
    ("UNH", "UnitedHealth"),
    ("JNJ", "Johnson & Johnson"),
    ("ABBV", "AbbVie"),
    ("LLY", "Eli Lilly"),
    ("MRK", "Merck"),
    ("PFE", "Pfizer"),
    # energy
    ("XOM", "Exxon Mobil"),
    ("CVX", "Chevron"),
    ("COP", "ConocoPhillips"),
]

# market code -> (label, stock list). The app builds one tab per market.
MARKETS = {
    "SG": ("🇸🇬 Singapore", SG_STOCKS),
    "US": ("🇺🇸 United States", US_STOCKS),
}

ALL_STOCKS = [(code, t, n) for code, (_, lst) in MARKETS.items() for (t, n) in lst]

# ticker -> sub-sector, for the Sector Rotation view. ^STI (the index) is omitted
# so the benchmark isn't rolled into a sector. Labels are shared across markets
# where they mean the same thing (Banks, Semiconductors, Insurance, ...).
SECTORS = {
    # --- SG ---
    "D05.SI": "Banks", "O39.SI": "Banks", "U11.SI": "Banks",
    "S68.SI": "Financial Services", "G07.SI": "Financial Services", "AIY.SI": "Financial Services",
    "Z74.SI": "Telecom", "CC3.SI": "Telecom", "CJLU.SI": "Telecom",
    "C6L.SI": "Airlines & Aviation", "S58.SI": "Airlines & Aviation", "S59.SI": "Airlines & Aviation",
    "C52.SI": "Land Transport", "S61.SI": "Land Transport",
    "9CI.SI": "Property Developers", "C09.SI": "Property Developers", "U14.SI": "Property Developers",
    "F17.SI": "Property Developers", "U06.SI": "Property Developers", "H78.SI": "Property Developers",
    "TQ5.SI": "Property Developers", "Z25.SI": "Property Developers",
    "A17U.SI": "Industrial REITs", "M44U.SI": "Industrial REITs", "ME8U.SI": "Industrial REITs",
    "BUOU.SI": "Industrial REITs", "J91U.SI": "Industrial REITs", "O5RU.SI": "Industrial REITs",
    "C38U.SI": "Retail REITs", "CRPU.SI": "Retail REITs", "J69U.SI": "Retail REITs",
    "AU8U.SI": "Retail REITs", "SK6U.SI": "Retail REITs", "JYEU.SI": "Retail REITs",
    "N2IU.SI": "Office REITs", "K71U.SI": "Office REITs", "T82U.SI": "Office REITs",
    "AJBU.SI": "Data-Centre REITs", "DCRU.SI": "Data-Centre REITs",
    "HMN.SI": "Hospitality REITs", "Q5T.SI": "Hospitality REITs", "J85.SI": "Hospitality REITs",
    "C2PU.SI": "Healthcare REITs",
    "BN4.SI": "Industrial Conglomerates", "J36.SI": "Industrial Conglomerates",
    "S63.SI": "Aerospace & Defence",
    "BS6.SI": "Marine & Offshore", "5E2.SI": "Marine & Offshore",
    "S08.SI": "Logistics",
    "U96.SI": "Utilities",
    "F34.SI": "Agriculture", "VC2.SI": "Agriculture", "584.SI": "Agriculture",
    "P8Z.SI": "Agriculture", "E5H.SI": "Agriculture",
    "D01.SI": "Consumer Staples", "Y92.SI": "Consumer Staples", "OV8.SI": "Consumer Staples",
    "C07.SI": "Consumer Discretionary", "G13.SI": "Consumer Discretionary",
    "V03.SI": "Technology Hardware", "AWX.SI": "Semiconductors",
    "Q0F.SI": "Healthcare", "BVA.SI": "Healthcare", "BSL.SI": "Healthcare",
    # --- US ---
    "MSFT": "Software", "ORCL": "Software", "CRM": "Software", "ADBE": "Software",
    "NOW": "Software", "INTU": "Software",
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "AMD": "Semiconductors",
    "INTC": "Semiconductors", "QCOM": "Semiconductors", "TXN": "Semiconductors", "MU": "Semiconductors",
    "GOOGL": "Internet & Media", "META": "Internet & Media", "NFLX": "Internet & Media",
    "DIS": "Internet & Media", "CMCSA": "Internet & Media",
    "AAPL": "Technology Hardware", "DELL": "Technology Hardware",
    "JPM": "Banks", "BAC": "Banks", "WFC": "Banks", "C": "Banks", "GS": "Banks", "MS": "Banks",
    "V": "Payments & Financial Services", "MA": "Payments & Financial Services",
    "AXP": "Payments & Financial Services", "PYPL": "Payments & Financial Services",
    "BRK-B": "Insurance", "PGR": "Insurance", "CB": "Insurance",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "MDLZ": "Consumer Staples",
    "CL": "Consumer Staples", "PM": "Consumer Staples",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    "MCD": "Restaurants", "SBUX": "Restaurants", "CMG": "Restaurants",
    "TSLA": "Autos", "GM": "Autos", "F": "Autos",
    "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare",
    "LLY": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
}
