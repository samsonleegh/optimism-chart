"""SG stock universe (SGX tickers, Yahoo Finance `.SI` suffix).

Default = STI constituents + a few large/liquid names. yfinance is the data
source, so this is the practical "all SG stocks" set: liquid counters with a
clean 10-year history. Add more (ticker, name) pairs to widen coverage.
"""

SG_STOCKS = [
    ("^STI", "STI Index"),                 # the Straits Times Index itself (benchmark)
    ("D05.SI", "DBS Group"),
    ("O39.SI", "OCBC"),
    ("U11.SI", "UOB"),
    ("Z74.SI", "Singtel"),
    ("S68.SI", "SGX"),
    ("C6L.SI", "Singapore Airlines"),
    ("9CI.SI", "CapitaLand Investment"),
    ("BN4.SI", "Keppel"),
    ("U96.SI", "Sembcorp Industries"),
    ("S63.SI", "ST Engineering"),
    ("F34.SI", "Wilmar International"),
    ("BS6.SI", "Yangzijiang Shipbuilding"),
    ("C07.SI", "Jardine Cycle & Carriage"),
    ("J36.SI", "Jardine Matheson"),
    ("H78.SI", "Hongkong Land"),
    ("D01.SI", "DFI Retail Group"),
    ("G13.SI", "Genting Singapore"),
    ("Y92.SI", "Thai Beverage"),
    ("C09.SI", "City Developments"),
    ("U14.SI", "UOL Group"),
    ("V03.SI", "Venture Corp"),
    ("S58.SI", "SATS"),
    ("C38U.SI", "CapitaLand Integrated Commercial Trust"),
    ("A17U.SI", "CapitaLand Ascendas REIT"),
    ("M44U.SI", "Mapletree Logistics Trust"),
    ("ME8U.SI", "Mapletree Industrial Trust"),
    ("N2IU.SI", "Mapletree Pan Asia Commercial Trust"),
    ("AJBU.SI", "Keppel DC REIT"),
    ("BUOU.SI", "Frasers Logistics & Commercial Trust"),
    ("T82U.SI", "Suntec REIT"),
    # --- extending toward top ~50 by market cap / liquidity ---
    ("Q0F.SI", "IHH Healthcare"),
    ("BVA.SI", "Top Glove"),
    ("S59.SI", "SIA Engineering"),
    ("C52.SI", "ComfortDelGro"),
    ("O32.SI", "Olam Group"),
    ("F17.SI", "GuocoLand"),
    ("U06.SI", "Singapore Land Group"),
    ("S08.SI", "Singapore Post"),
    ("BSL.SI", "Raffles Medical"),
    ("584.SI", "First Resources"),
    ("P8Z.SI", "Bumitama Agri"),
    ("5E2.SI", "Seatrium"),
    ("AWX.SI", "AEM Holdings"),
    ("5CP.SI", "Silverlake Axis"),
    ("OV8.SI", "Sheng Siong Group"),
    ("CC3.SI", "StarHub"),
    ("CRPU.SI", "Sasseur REIT"),
    ("HMN.SI", "CapitaLand Ascott Trust"),
    ("J91U.SI", "ESR-REIT"),
    ("K71U.SI", "Keppel REIT"),
]

US_STOCKS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet"),
    ("META", "Meta Platforms"),
    ("AVGO", "Broadcom"),
    ("TSLA", "Tesla"),
    ("BRK-B", "Berkshire Hathaway"),
    ("JPM", "JPMorgan Chase"),
    ("V", "Visa"),
    ("MA", "Mastercard"),
    ("WMT", "Walmart"),
    ("UNH", "UnitedHealth"),
    ("XOM", "Exxon Mobil"),
    ("JNJ", "Johnson & Johnson"),
    ("PG", "Procter & Gamble"),
    ("HD", "Home Depot"),
    ("COST", "Costco"),
    ("ORCL", "Oracle"),
    ("KO", "Coca-Cola"),
    ("PEP", "PepsiCo"),
    ("ABBV", "AbbVie"),
    ("BAC", "Bank of America"),
    ("CRM", "Salesforce"),
    ("MCD", "McDonald's"),
    ("NFLX", "Netflix"),
    ("ADBE", "Adobe"),
    ("AMD", "Advanced Micro Devices"),
    ("DIS", "Walt Disney"),
]

# market code -> (label, stock list). The app builds one tab per market.
MARKETS = {
    "SG": ("🇸🇬 Singapore", SG_STOCKS),
    "US": ("🇺🇸 United States", US_STOCKS),
}

ALL_STOCKS = [(code, t, n) for code, (_, lst) in MARKETS.items() for (t, n) in lst]

# ticker -> sector, for the Sector Rotation view. ^STI (the index) is intentionally
# omitted so the benchmark isn't rolled into a sector. Keep labels shared across
# markets (Banks, Technology, ...) so the taxonomy reads the same on both boards.
SECTORS = {
    # --- SG ---
    "D05.SI": "Banks", "O39.SI": "Banks", "U11.SI": "Banks",
    "Z74.SI": "Telecom", "CC3.SI": "Telecom",
    "S68.SI": "Financial Services",
    "C6L.SI": "Transport", "S58.SI": "Transport", "C52.SI": "Transport",
    "9CI.SI": "Real Estate", "C09.SI": "Real Estate", "U14.SI": "Real Estate",
    "F17.SI": "Real Estate", "U06.SI": "Real Estate", "H78.SI": "Real Estate",
    "BN4.SI": "Industrials", "S63.SI": "Industrials", "BS6.SI": "Industrials",
    "5E2.SI": "Industrials", "S59.SI": "Industrials", "S08.SI": "Industrials",
    "U96.SI": "Utilities",
    "F34.SI": "Agriculture", "O32.SI": "Agriculture", "584.SI": "Agriculture",
    "P8Z.SI": "Agriculture",
    "C07.SI": "Consumer Discretionary", "G13.SI": "Consumer Discretionary",
    "J36.SI": "Conglomerates",
    "D01.SI": "Consumer Staples", "Y92.SI": "Consumer Staples", "OV8.SI": "Consumer Staples",
    "V03.SI": "Technology", "5CP.SI": "Technology", "AWX.SI": "Semiconductors",
    "Q0F.SI": "Healthcare", "BVA.SI": "Healthcare", "BSL.SI": "Healthcare",
    "C38U.SI": "REITs", "A17U.SI": "REITs", "M44U.SI": "REITs", "ME8U.SI": "REITs",
    "N2IU.SI": "REITs", "AJBU.SI": "REITs", "BUOU.SI": "REITs", "T82U.SI": "REITs",
    "CRPU.SI": "REITs", "HMN.SI": "REITs", "J91U.SI": "REITs", "K71U.SI": "REITs",
    # --- US ---
    "AAPL": "Technology", "MSFT": "Technology", "ORCL": "Technology",
    "CRM": "Technology", "ADBE": "Technology",
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "AMD": "Semiconductors",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "GOOGL": "Communication/Media", "META": "Communication/Media",
    "NFLX": "Communication/Media", "DIS": "Communication/Media",
    "JPM": "Banks", "BAC": "Banks",
    "BRK-B": "Financial Services", "V": "Financial Services", "MA": "Financial Services",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare",
    "XOM": "Energy",
}
