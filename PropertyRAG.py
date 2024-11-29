import re
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from fuzzywuzzy import fuzz


class PropertyRAG:
    def __init__(self, df, text_model, numeric_model, tfidf_vectorizer):
        self.df = df
        self.text_model = text_model
        self.numeric_model = numeric_model
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword = StopWordRemoverFactory().create_stop_word_remover()
        self.tfidf_vectorizer = tfidf_vectorizer

        # Prepare combined text features
        self.df["text_combined"] = (
            df["Judul_Clean"] + " " + df["Lokasi_Clean"] + " " + df["Deskripsi_Clean"]
        )

        # Generate TF-IDF features for the dataset
        self.tfidf_matrix = self.tfidf_vectorizer.transform(
            self.df["text_combined"].fillna("").astype(str)
        )

        # Prepare numeric features (same as in training)
        self.numeric_features = np.column_stack(
            [
                df[
                    [
                        "Kamar_Normalized",
                        "WC_Normalized",
                        "Parkir_Normalized",
                        "Luas_Tanah_Normalized",
                        "Luas_Bangunan_Normalized",
                    ]
                ].values,
                df["Harga_Normalized"] / df["Luas_Bangunan_Normalized"],
                df["Luas_Bangunan_Normalized"] / df["Luas_Tanah_Normalized"],
                df["Kamar_Normalized"] * df["WC_Normalized"],
            ]
        )

        # Pemetaan lokasi dan fasilitas
        self.location_mapping = {
            "jogja": ["yogyakarta", "jogjakarta", "yogya"],
            "bantul": [
                "bambanglipuro",
                "banguntapan",
                "bantul",
                "dlingo",
                "imogiri",
                "jetis bantul",
                "kasihan",
                "kretek",
                "pajangan",
                "pandak",
                "piyungan",
                "pleret",
                "pundong",
                "sanden",
                "dayu",
                "sewon",
                "srandakan",
            ],
            "gunung kidul": [
                "gedangsari",
                "girisubo",
                "karangmojo",
                "ngawen",
                "nglipar",
                "paliyan",
                "panggang",
                "patuk",
                "playen",
                "ponjong",
                "purwosari",
                "rongkop",
                "saptosari",
                "semanu",
                "semin",
                "tanjungsari",
                "tepus",
                "wonosari",
            ],
            "kulon progo": [
                "galur",
                "girimulyo",
                "kalibawang",
                "kokap",
                "lendah",
                "nanggulan",
                "panjatan",
                "pengasih",
                "samigaluh",
                "sentolo",
                "temon",
                "wates",
            ],
            "sleman": [
                "berbah",
                "cangkringan",
                "depok",
                "gamping",
                "godean",
                "kalasan",
                "minggir",
                "mlati",
                "moyudan",
                "ngaglik",
                "ngemplak",
                "pakem",
                "prambanan",
                "seyegan",
                "sleman",
                "tempel",
                "turi",
            ],
            "yogyakarta": [
                "danurejan",
                "gedongtengen",
                "gondokusuman",
                "gondomanan",
                "jetis",
                "kotagede",
                "kraton",
                "mantrijeron",
                "mergangsan",
                "ngampilan",
                "pakualaman",
                "tegalrejo",
                "umbulharjo",
                "wirobrajan",
            ],
        }

    def preprocess_query(self, query):
        try:
            query = query.lower()
            query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
            query = re.sub(r"\s+", " ", query).strip()
            query = self.stemmer.stem(query)
            query = self.stopword.remove(query)
            return query
        except Exception as e:
            logging.error(f"Query preprocessing error: {e}")
            return query  # Return original query if preprocessing fails

    def extract_numeric_requirements(self, query):
        # Pola regex untuk kamar - menambahkan beberapa variasi umum
        kamar_patterns = [
            r"(\d+)\s*(?:kamar|ruang\s*tidur|bedroom|kmr|bed\s*room|kamar\s*tidur|kt\b)",
            r"(?:mau|ingin|cari|butuh|perlu)\s*(?:rumah\s*(?:dengan|yang|ada))?\s*(\d+)\s*(?:kamar|bedroom)",
            r"(?:dengan|ada|memiliki)\s*(\d+)\s*(?:kamar|ruang\s*tidur)",
            r"(\d+)(?:kamar|ruang\s*tidur|bedroom)",
            r"kamar(?:nya|tidur)?\s*(\d+)\s*(?:buah)?",
        ]

        # Pola regex untuk kamar mandi/WC - menambahkan variasi
        wc_patterns = [
            r"(\d+)\s*(?:kamar\s*mandi|wc|toilet|bathroom|km|kmandi)",
            r"(?:dengan|ada|memiliki)\s*(?:kamar\s*mandi|wc|toilet)?\s*(\d+)",
            r"(\d+)(?:kamar\s*mandi|wc|toilet|km)",
            r"(?:toilet|km|kamar\s*mandi)(?:nya)?\s*(\d+)",
            r"k(?:amar)?\.?\s*m(?:andi)?\s*(\d+)",
        ]

        # Pola regex untuk parkir - menambahkan variasi
        parkir_patterns = [
            r"(\d+)\s*(?:parkir|park|tempat\s*parkir|slot|area\s*parkir|mobil)",
            r"(?:parkiran|slot\s*parkir|kapasitas\s*parkir)\s*(\d+)",
            r"(?:muat|untuk)\s*(\d+)\s*(?:mobil|kendaraan|motor)",
            r"parkir\s*(?:untuk|buat)?\s*(\d+)\s*(?:mobil|kendaraan)?",
            r"(?:bisa\s*)?parkir\s*(\d+)\s*mobil",
        ]

        # Pola regex yang diperbaiki untuk harga
        price_patterns = [
            # Pola dengan kata kunci harga/budget/dana
            r"(?:harga|budget|dana|biaya|harganya|seharga|kisaran)\s*(?:sekitar|kurang\s*dari|dibawah|sampai|maksimal|max)?\s*(?:rp\.?\s*)?(\d+(?:[,.]\d+)?)\s*(?:milyar|miliar|milyard|m\b)(?:\s*(?:rupiah|rp))?",
            r"(?:harga|budget|dana|biaya|harganya|seharga|kisaran)\s*(?:sekitar|kurang\s*dari|dibawah|sampai|maksimal|max)?\s*(?:rp\.?\s*)?(\d+(?:[,.]\d+)?)\s*(?:juta|jt|j\b)(?:\s*(?:rupiah|rp))?",
            # Pola dengan angka didahului indikator maksimum
            r"(?:dibawah|di\s*bawah|maksimal|max|maksimum|paling\s*mahal|hingga|sampai)\s*(?:rp\.?\s*)?(\d+(?:[,.]\d+)?)\s*(?:milyar|miliar|milyard|m\b)(?:\s*(?:rupiah|rp))?",
            r"(?:dibawah|di\s*bawah|maksimal|max|maksimum|paling\s*mahal|hingga|sampai)\s*(?:rp\.?\s*)?(\d+(?:[,.]\d+)?)\s*(?:juta|jt|j\b)(?:\s*(?:rupiah|rp))?",
            # Pola dengan Rp didepan
            r"(?:rp\.?\s*)?(\d+(?:[,.]\d+)?)\s*(?:milyar|miliar|milyard|m\b)(?:\s*(?:rupiah|rp))?",
            r"(?:rp\.?\s*)?(\d+(?:[,.]\d+)?)\s*(?:juta|jt|j\b)(?:\s*(?:rupiah|rp))?",
        ]
        # Add new patterns for luas tanah
        luas_tanah_patterns = [
            r"(?:luas|lt|land\s*size)\s*(?:tanah)?\s*(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)",
            r"(?:tanah(?:nya)?|lot)\s*(?:seluas|ukuran)?\s*(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)",
            r"(?:tanah|lot)\s*(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)",
            r"(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)\s*(?:tanah|lot)",
        ]

        # Add new patterns for luas bangunan
        luas_bangunan_patterns = [
            r"(?:luas|lb|building\s*size)\s*(?:bangunan)?\s*(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)",
            r"(?:bangunan(?:nya)?)\s*(?:seluas|ukuran)?\s*(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)",
            r"(?:bangunan)\s*(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)",
            r"(\d+(?:[,.]\d+)?)\s*(?:m2|meter|m²)\s*(?:bangunan)",
        ]

        def normalize_price_string(price_str):
            # Membersihkan string harga
            price_str = price_str.replace(".", "")
            price_str = price_str.replace(",", ".")
            try:
                return float(price_str)
            except ValueError:
                return 0.0

        # Ekstraksi harga dengan pengecekan unit yang lebih ketat
        max_price = None  # Default None jika tidak ada harga yang disebutkan
        for pattern in price_patterns:
            price_matches = re.findall(pattern, query.lower(), re.IGNORECASE)
            if price_matches:
                price_str = price_matches[0]
                price_value = normalize_price_string(price_str)

                # Deteksi unit dengan lebih spesifik
                price_context = query.lower()
                if any(
                    unit
                    in price_context[
                        price_context.find(str(price_str)) : price_context.find(
                            str(price_str)
                        )
                        + 20
                    ]
                    for unit in ["milyar", "miliar", "milyard", "m"]
                ):
                    max_price = price_value * 1e9
                elif any(
                    unit
                    in price_context[
                        price_context.find(str(price_str)) : price_context.find(
                            str(price_str)
                        )
                        + 20
                    ]
                    for unit in ["juta", "jt", "j"]
                ):
                    max_price = price_value * 1e6
                break

        def find_first_match(patterns, text, default=None):
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        value = matches[0]
                        if isinstance(value, tuple):
                            value = value[0]
                        return int(float(value))
                    except (ValueError, IndexError):
                        continue
            return default

        # Ekstraksi nilai numerik
        kamar = find_first_match(kamar_patterns, query)
        wc = find_first_match(wc_patterns, query)
        parkir = find_first_match(parkir_patterns, query)
        luas_tanah = find_first_match(luas_tanah_patterns, query)
        luas_bangunan = find_first_match(luas_bangunan_patterns, query)

        # Update the return statement to include new values
        return kamar, wc, parkir, max_price, luas_tanah, luas_bangunan

    def find_location(self, query):
        locations = list(
            set(
                [
                    "sleman",
                    "bantul",
                    "yogyakarta",
                    "jogja",
                    "kulon progo",
                    "gunung kidul",
                ]
                + [
                    loc
                    for variants in self.location_mapping.values()
                    for loc in variants
                ]
            )
        )
        query_location = None
        max_ratio = 0
        for loc in locations:
            ratio = fuzz.partial_ratio(loc, query.lower())
            if ratio > max_ratio and ratio > 80:
                max_ratio = ratio
                query_location = loc
        return query_location

    def get_model_predictions(self, query_text, numeric_features):
        # Generate text features using TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([query_text]).toarray()

        # Get predictions from both models
        text_pred = self.text_model.predict(query_tfidf)

        # Sesuaikan dengan 9 fitur seperti saat training
        numeric_input = np.array(
            [
                numeric_features[0],  # Kamar_Normalized
                numeric_features[1],  # WC_Normalized
                numeric_features[2],  # Parkir_Normalized
                numeric_features[3],  # Luas_Tanah_Normalized
                numeric_features[4],  # Luas_Bangunan_Normalized
                0,  # Harga_Normalized (bisa diisi 0 atau nilai default)
                numeric_features[5],  # Harga/Luas_Bangunan
                numeric_features[6],  # Luas_Bangunan/Luas_Tanah
                numeric_features[7],  # Kamar * WC
            ]
        )

        numeric_pred = self.numeric_model.predict(numeric_input.reshape(1, -1))

        # Combine predictions (you can adjust the weights)
        combined_pred = 0.2 * text_pred + 0.8 * numeric_pred
        return combined_pred

    def get_recommendations(self, query, top_k=5):
        clean_query = self.preprocess_query(query)
        kamar, wc, parkir, max_price, luas_tanah, luas_bangunan = (
            self.extract_numeric_requirements(query)
        )
        query_location = self.find_location(query)

        # Calculate query_numeric_features dengan nilai default
        query_numeric_features = np.array(
            [
                kamar / 5 if kamar is not None else 0.5,  # Default value 0.5
                wc / 3 if wc is not None else 0.5,
                parkir / 2 if parkir is not None else 0.5,
                (
                    luas_tanah / 150 if luas_tanah is not None else 0.5
                ),  # Luas_Tanah_Normalized
                (
                    luas_bangunan / 150 if luas_bangunan is not None else 0.5
                ),  # Luas_Bangunan_Normalized
                0.5,  # Harga/Luas_Bangunan
                (
                    luas_bangunan / luas_tanah
                    if (luas_bangunan is not None and luas_tanah is not None)
                    else 0.5
                ),  # Luas_Bangunan/Luas_Tanah
                (kamar * wc / 15) if (kamar is not None and wc is not None) else 0.5,
            ]
        )

        # Hitung predicted_price di awal
        predicted_price = self.get_model_predictions(
            clean_query, query_numeric_features
        )

        # Filter DataFrame based on exact requirements first
        exact_matches_df = self.df.copy()
        exact_matches_df["relevance_score"] = 0.0

        # Apply exact filters only if the values exist
        if max_price is not None and max_price != float("inf"):
            exact_matches_df = exact_matches_df[exact_matches_df["Harga"] <= max_price]

        if query_location:
            location_mask = (
                exact_matches_df["Lokasi_Clean"]
                .fillna("")
                .astype(str)
                .apply(lambda x: fuzz.partial_ratio(query_location, x.lower()) >= 90)
            )
            exact_matches_df = exact_matches_df[location_mask]

        # Apply exact numeric filters only if values exist
        if kamar is not None:
            exact_matches_df = exact_matches_df[exact_matches_df["Kamar"] == kamar]

        if wc is not None:
            exact_matches_df = exact_matches_df[exact_matches_df["WC"] == wc]

        if parkir is not None:
            exact_matches_df = exact_matches_df[exact_matches_df["Parkir"] == parkir]

        if luas_tanah is not None:
            exact_matches_df = exact_matches_df[
                exact_matches_df["Luas_Tanah"] == luas_tanah
            ]

        if luas_bangunan is not None:
            exact_matches_df = exact_matches_df[
                exact_matches_df["Luas_Bangunan"] == luas_bangunan
            ]

        # Process exact matches
        if len(exact_matches_df) > 0:
            exact_tfidf_matrix = self.tfidf_vectorizer.transform(
                exact_matches_df["text_combined"].fillna("").astype(str)
            )
            exact_text_similarities = cosine_similarity(
                self.tfidf_vectorizer.transform([clean_query]), exact_tfidf_matrix
            ).flatten()

            exact_price_diff_scores = 1 / (
                1
                + np.abs(
                    exact_matches_df["Harga_Normalized"].values
                    - predicted_price.flatten()[0]
                )
            )

            exact_matches_df["relevance_score"] = (
                0.2 * exact_text_similarities
                + 0.7 * exact_price_diff_scores
                + 0.1 * np.zeros(len(exact_matches_df))
            )
            exact_matches_df = exact_matches_df.sort_values(
                "relevance_score", ascending=False
            )

        # Proses flexible recommendations
        flexible_df = self.df.copy()
        flexible_df["relevance_score"] = 0.0

        # Apply flexible filters
        if max_price is not None and max_price != float("inf"):
            flexible_df = flexible_df[
                flexible_df["Harga"] <= max_price * 1.2
            ]  # 20% tolerance

        if query_location:
            location_mask = (
                flexible_df["Lokasi_Clean"]
                .fillna("")
                .astype(str)
                .apply(lambda x: fuzz.partial_ratio(query_location, x.lower()) >= 70)
            )
            flexible_df = flexible_df[location_mask]

        # Apply numeric filters with tolerance only if values exist
        if kamar is not None:
            flexible_df = flexible_df[
                (flexible_df["Kamar"] >= max(1, kamar - 1))
                & (flexible_df["Kamar"] <= kamar + 1)
            ]

        if wc is not None:
            flexible_df = flexible_df[
                (flexible_df["WC"] >= max(1, wc - 1)) & (flexible_df["WC"] <= wc + 1)
            ]

        if parkir is not None:
            flexible_df = flexible_df[
                (flexible_df["Parkir"] >= max(1, parkir - 1))
                & (flexible_df["Parkir"] <= parkir + 1)
            ]

        if luas_tanah is not None:
            flexible_df = flexible_df[
                (flexible_df["Luas_Tanah"] >= max(1, luas_tanah - 10))
                & (flexible_df["Luas_Tanah"] <= luas_tanah + 10)
            ]

        if luas_bangunan is not None:
            flexible_df = flexible_df[
                (flexible_df["Luas_Bangunan"] >= max(1, luas_bangunan - 10))
                & (flexible_df["Luas_Bangunan"] <= luas_bangunan + 10)
            ]

        # Remove exact matches from flexible recommendations
        if len(exact_matches_df) > 0:
            flexible_df = flexible_df[~flexible_df.index.isin(exact_matches_df.index)]

        # Process flexible matches if any remain
        if len(flexible_df) > 0:
            flexible_tfidf_matrix = self.tfidf_vectorizer.transform(
                flexible_df["text_combined"].fillna("").astype(str)
            )
            flexible_text_similarities = cosine_similarity(
                self.tfidf_vectorizer.transform([clean_query]), flexible_tfidf_matrix
            ).flatten()

            flexible_price_diff_scores = 1 / (
                1
                + np.abs(
                    flexible_df["Harga_Normalized"].values
                    - predicted_price.flatten()[0]
                )
            )

            flexible_df["relevance_score"] = (
                0.2 * flexible_text_similarities
                + 0.7 * flexible_price_diff_scores
                + 0.1 * np.zeros(len(flexible_df))
            )
            flexible_df = flexible_df.sort_values("relevance_score", ascending=False)

        # Return empty DataFrame if no matches found
        if len(exact_matches_df) == 0 and len(flexible_df) == 0:
            return pd.DataFrame()

        # Combine results with Exact Matches First
        final_recommendations = exact_matches_df

        # Add flexible matches if needed to reach top_k
        if len(final_recommendations) < top_k and len(flexible_df) > 0:
            remaining_slots = top_k - len(final_recommendations)
            final_recommendations = pd.concat(
                [final_recommendations, flexible_df.head(remaining_slots)]
            )

        return final_recommendations[
            [
                "Judul",
                "Lokasi",
                "Harga",
                "Kamar",
                "WC",
                "Parkir",
                "Luas_Tanah",
                "Luas_Bangunan",
                "Image_Link",
                "Property_Link",
                "relevance_score",
            ]
        ]
