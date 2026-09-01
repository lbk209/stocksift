import requests
import pandas as pd
from tqdm.auto import tqdm


def fetch_naver_ohlcv(tickers, start, end=None, errors="ignore"):
    """
    Fetch daily OHLCV data for Korean stocks from Naver Finance.

    Parameters
    ----------
    tickers : list[str]
        Stock ticker codes, e.g. ["005930", "000660"].
    start : str or datetime-like
        Start date.
    end : str or datetime-like, optional
        End date. Defaults to today.
    errors : {"ignore", "raise"}, default "ignore"
        - "ignore": skip ticker-level data failures and continue.
        - "raise": stop immediately on the first ticker-level failure.

    Returns
    -------
    pandas.DataFrame
        MultiIndex: (date, ticker)
        Columns: open, high, low, close, volume

    Notes
    -----
    HTTP/API failures stop the function regardless of `errors`.
    """
    if errors not in {"ignore", "raise"}:
        raise ValueError("errors must be 'ignore' or 'raise'")

    if not tickers:
        raise ValueError("tickers must not be empty")

    start = pd.Timestamp(start).normalize()
    end = (
        pd.Timestamp.today().normalize()
        if end is None
        else pd.Timestamp(end).normalize()
    )

    if start > end:
        raise ValueError("start must be on or before end")

    raw_cols = {
        "localTradedAt": "date",
        "openPrice": "open",
        "highPrice": "high",
        "lowPrice": "low",
        "closePrice": "close",
        "accumulatedTradingVolume": "volume",
    }
    cols = ["open", "high", "low", "close", "volume"]

    frames = []
    failed_tickers = []

    for ticker in tqdm(tickers, desc="Fetching OHLCV", unit="ticker"):
        ticker = str(ticker)

        try:
            url = f"https://m.stock.naver.com/api/stock/{ticker}/price"

            pages = []
            page = 1

            while True:
                params = {
                    "pageSize": 20,
                    "page": page,
                }

                response = requests.get(url, params=params)

                # HTTP/API errors should never be silently ignored.
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise RuntimeError(
                        f"Naver Finance API error "
                        f"({response.status_code}): {response.url}"
                    ) from exc

                data = response.json()

                if not isinstance(data, list):
                    raise RuntimeError(
                        "Unexpected Naver Finance API response format"
                    )

                if not data:
                    break

                df_page = pd.DataFrame(data)

                missing = set(raw_cols) - set(df_page.columns)
                if missing:
                    raise RuntimeError(
                        f"Naver Finance API fields changed: "
                        f"missing {sorted(missing)}"
                    )

                df_page = (
                    df_page[list(raw_cols)]
                    .rename(columns=raw_cols)
                )

                df_page["date"] = pd.to_datetime(df_page["date"])

                pages.append(df_page)

                # Data are returned from newest to oldest.
                if df_page["date"].min() <= start:
                    break

                page += 1

            if not pages:
                raise ValueError("no price data returned")

            df = pd.concat(pages, ignore_index=True)

            df = df[
                (df["date"] >= start)
                & (df["date"] <= end)
            ].copy()

            if df.empty:
                raise ValueError("no price data in requested date range")

            for col in cols:
                if df[col].dtype == object:
                    df[col] = df[col].str.replace(",", "", regex=False)

                df[col] = pd.to_numeric(df[col], errors="raise")

            df["ticker"] = ticker

            frames.append(
                df.set_index(["date", "ticker"])[cols]
            )

        except RuntimeError:
            # API-level problem: always stop.
            raise

        except Exception as exc:
            if errors == "raise":
                raise RuntimeError(
                    f"failed to fetch {ticker}"
                ) from exc

            failed_tickers.append(ticker)

    if len(failed_tickers) <= 3:
        for ticker in failed_tickers:
            print(f"WARNING: failed to fetch {ticker}")
    elif failed_tickers:
        print(f"WARNING: failed to fetch {len(failed_tickers)} tickers")

    if not frames:
        index = pd.MultiIndex.from_arrays(
            [[], []],
            names=["date", "ticker"],
        )
        return pd.DataFrame(index=index, columns=cols)

    return pd.concat(frames).sort_index()