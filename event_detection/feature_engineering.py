import pandas as pd
import numpy as np
from itertools import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class Features:
    def __init__(self):
        self.X = pd.DataFrame(data=None, columns=["body_length"])

    def features_clean(self, df):
        self.X["object_id"] = df["object_id"]
        self.X["body_length"] = df["body_length"]
        self.X["name_length"] = df["name_length"]
        self.X["sale_duration"] = df["sale_duration"].fillna(df["sale_duration"].mean())
        self.X["user_age"] = df["user_age"]
        self.X["fb_published"] = df["fb_published"]
        self.X["has_logo"] = df["has_logo"]
        self.X["has_analytics"] = df["has_analytics"]
        self.X["org_facebook"] = df["org_facebook"].fillna(df["org_facebook"].mean())
        self.X["org_twitter"] = df["org_twitter"].fillna(df["org_twitter"].mean())

        selected_dummies = [
            "country",
            "venue_state",
            "currency",
            "delivery_method",
            "payout_type",
        ]
        self.X = self.dummies(df.loc[:, selected_dummies])
        self.X = self.ticket_types(df.loc[:, "ticket_types"])
        self.X = self.X.drop("object_id", axis=1)

        return self.X

    def ticket_types(self, tickets):

        event_types = pd.DataFrame(list(chain.from_iterable(tickets)))

        totals = event_types.groupby("event_id").sum()
        avg_cost = (
            event_types.groupby(["event_id", "availability", "quantity_total"])
            .sum()
            .groupby("event_id")
            .mean()
        )
        unique_event_ids = pd.merge(totals, avg_cost, on=["event_id"]).reset_index()
        unique_event_ids = unique_event_ids.drop(
            columns=["cost_x", "availability"]
        ).rename(
            index=str,
            columns={
                "event_id": "object_id",
                "cost_y": "avg_ticket_cost",
                "quantity_total": "tot_ticket_quant",
            },
        )
        self.X = pd.merge(self.X, unique_event_ids, on="object_id", how="outer")

        self.X["ticket_type_missing"] = self.X["avg_ticket_cost"].where(
            self.X["avg_ticket_cost"].isnull(), 0
        )
        self.X["ticket_type_missing"] = self.X["ticket_type_missing"].where(
            self.X["ticket_type_missing"] == 0, 1
        )

        self.X["avg_ticket_cost"] = self.X["avg_ticket_cost"].fillna(
            self.X["avg_ticket_cost"].mean()
        )
        self.X["tot_ticket_quant"] = self.X["tot_ticket_quant"].fillna(
            self.X["tot_ticket_quant"].mean()
        )

        return self.X

    def dummies(self, df):

        self.X["AUD"] = df["currency"].where(df["currency"] == "AUD", 0)
        self.X["AUD"] = self.X["AUD"].where(self.X["AUD"] == 0, 1)

        self.X["CAD"] = df["currency"].where(df["currency"] == "CAD", 0)
        self.X["CAD"] = self.X["CAD"].where(self.X["CAD"] == 0, 1)

        self.X["EUR"] = df["currency"].where(df["currency"] == "EUR", 0)
        self.X["EUR"] = self.X["EUR"].where(self.X["EUR"] == 0, 1)

        self.X["GBP"] = df["currency"].where(df["currency"] == "GBP", 0)
        self.X["GBP"] = self.X["GBP"].where(self.X["GBP"] == 0, 1)

        self.X["MXN"] = df["currency"].where(df["currency"] == "MXN", 0)
        self.X["MXN"] = self.X["MXN"].where(self.X["MXN"] == 0, 1)

        self.X["NZD"] = df["currency"].where(df["currency"] == "NZD", 0)
        self.X["NZD"] = self.X["NZD"].where(self.X["NZD"] == 0, 1)

        self.X["USD"] = df["currency"].where(df["currency"] == "USD", 0)
        self.X["USD"] = self.X["USD"].where(self.X["USD"] == 0, 1)

        self.X["deliv_method_0"] = df["delivery_method"].where(
            df["delivery_method"] == 0.0, 0
        )
        self.X["deliv_method_0"] = self.X["deliv_method_0"].where(
            self.X["deliv_method_0"] == 0, 1
        )

        self.X["deliv_method_1"] = df["delivery_method"].where(
            df["delivery_method"] == 1.0, 0
        )
        self.X["deliv_method_1"] = self.X["deliv_method_1"].where(
            self.X["deliv_method_1"] == 0, 1
        )

        self.X["deliv_method_3"] = df["delivery_method"].where(
            df["delivery_method"] == 3.0, 0
        )
        self.X["deliv_method_3"] = self.X["deliv_method_3"].where(
            self.X["deliv_method_3"] == 0, 1
        )

        self.X["payout_type_cash"] = df["payout_type"].where(
            df["payout_type"] == "ACH", 0
        )
        self.X["payout_type_cash"] = self.X["payout_type_cash"].where(
            self.X["payout_type_cash"] == 0, 1
        )

        self.X["payout_type_check"] = df["payout_type"].where(
            df["payout_type"] == "CHECK", 0
        )
        self.X["payout_type_check"] = self.X["payout_type_check"].where(
            self.X["payout_type_check"] == 0, 1
        )

        self.X["payout_type_missing"] = df["payout_type"].where(
            df["payout_type"] == np.nan, 0
        )
        self.X["payout_type_missing"] = self.X["payout_type_missing"].where(
            self.X["payout_type_missing"] == 0, 1
        )

        return self.X
