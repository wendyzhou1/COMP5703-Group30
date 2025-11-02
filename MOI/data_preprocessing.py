import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import sparse
import os
import warnings
import joblib

warnings.filterwarnings('ignore')


class MultiOmicsDataPreprocessor:
    """多组学数据预处理和保存"""

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.dna_scaler = StandardScaler()
        self.rna_scaler = StandardScaler()
        self.full_re = re.compile(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})[A-Z]?.*')

    # ------------- 条形码辅助 -------------
    def _extract_patient_id(self, s: str):
        if not isinstance(s, str):
            return None
        m = self.full_re.match(s.strip())
        return m.group(1) if m else None

    def _extract_sample_code(self, s: str):
        if not isinstance(s, str):
            return None
        m = self.full_re.match(s.strip())
        return m.group(2) if m else None

    def _is_primary_by_barcode(self, s: str) -> bool:
        code = self._extract_sample_code(s)
        return True if code is None else (code == "01")

    # ------------- 读文件 -------------
    def _read_tsv(self, filepath):
        try:
            return pd.read_csv(filepath, sep='\t', low_memory=False)
        except Exception:
            return pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)

    # ------------- 临床数据 -------------
    def load_clinical_data(self, clinical_file):
        df = self._read_tsv(clinical_file)
        tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
        if len(tcga_cols) >= 5:  # 宽表
            attr_col = df.columns[0]
            row_mask = df[attr_col].astype(str).str.contains(
                r'(subtype|GeneExp_Subtype|Subtype_mRNA|mRNA.*cluster|verhaak)',
                case=False, regex=True, na=False
            )
            if not row_mask.any():
                raise ValueError("在临床矩阵里找不到亚型行")
            row = df.loc[row_mask].iloc[0]
            sub_series = row.drop(labels=[attr_col])
            clin = pd.DataFrame({
                "sample_barcode": sub_series.index.astype(str),
                "subtype": sub_series.values
            })
        else:
            sid_col = None
            for col in ["sampleID", "bcr_sample_barcode", "sample", "barcode", "Sample", "SAMPLE_ID"]:
                if col in df.columns:
                    sid_col = col
                    break
            if sid_col is None:
                cand = [c for c in df.columns if df[c].astype(str).str.startswith("TCGA-").any()]
                sid_col = cand[0] if cand else df.columns[0]

            subtype_col = None
            for col in ["GeneExp_Subtype", "Subtype_mRNA", "Subtype", "subtype", "mRNA_cluster"]:
                if col in df.columns:
                    subtype_col = col
                    break
            if subtype_col is None:
                cands = [c for c in df.columns if ('subtype' in str(c).lower() or 'cluster' in str(c).lower())]
                if cands:
                    subtype_col = cands[0]
                else:
                    raise ValueError("临床数据中未找到亚型列（subtype/cluster 等）")

            clin = df[[sid_col, subtype_col]].copy()
            clin = clin.rename(columns={sid_col: "sample_barcode", subtype_col: "subtype"})

        def norm_subtype(x):
            if pd.isna(x):
                return x
            s = str(x).strip()
            return s[:1].upper() + s[1:].lower() if s else s

        clin["sample_barcode"] = clin["sample_barcode"].astype(str)
        clin["patient_id"] = clin["sample_barcode"].apply(self._extract_patient_id)
        clin["is_primary"] = clin["sample_barcode"].apply(self._is_primary_by_barcode)
        clin["subtype"] = clin["subtype"].apply(norm_subtype)

        clin = clin[clin["patient_id"].notna() & clin["subtype"].notna()]
        clin = (clin.sort_values(["patient_id", "is_primary"], ascending=[True, False])
                .drop_duplicates(subset=["patient_id"], keep="first"))

        out = clin[["patient_id", "subtype"]].rename(columns={"patient_id": "short_id"})
        print(f"临床数据（去重后病人数）: {len(out)}")
        return out

    # ------------- 组学数据 -------------
    def load_omics_data(self, omics_file, data_type="methylation"):
        df = self._read_tsv(omics_file)
        first_col_lower = str(df.columns[0]).strip().lower()
        if first_col_lower in {"gene", "genes", "hugo", "symbol", "gene symbol", "id", "probe", "probes",
                               "composite element ref"}:
            df = df.set_index(df.columns[0])

        def cols_have_tcga(_df):
            return any(str(c).startswith("TCGA-") for c in _df.columns)

        if not cols_have_tcga(df):
            first_col = df.columns[0]
            if df[first_col].astype(str).str.startswith("TCGA-").any():
                df = df.set_index(first_col).transpose()

        if not any(str(c).startswith("TCGA-") for c in df.columns):
            if any(str(i).startswith("TCGA-") for i in df.index):
                df = df.transpose()

        tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
        if len(tcga_cols) == 0:
            raise ValueError(f"{data_type}数据中未发现TCGA样本")

        df = df[tcga_cols]
        df = df.apply(pd.to_numeric, errors="coerce")

        X = df.transpose().copy()
        X.index.name = "sample_barcode"
        return X

    def load_mutation_data(self, mutation_file):
        return self.load_omics_data(mutation_file, "mutation")

    # ------------- 样本匹配 -------------
    def match_samples(self, omics_data: pd.DataFrame, clinical_data: pd.DataFrame):
        patient_ids = omics_data.index.to_series().astype(str).apply(self._extract_patient_id)
        keep = patient_ids.notna()
        omics_data = omics_data.loc[keep].copy()
        patient_ids = patient_ids.loc[keep]

        dedup_mask = ~patient_ids.duplicated()
        omics_data = omics_data.loc[dedup_mask]
        patient_ids = patient_ids.loc[dedup_mask]

        short_to_subtype = dict(zip(clinical_data["short_id"], clinical_data["subtype"]))
        y = patient_ids.map(short_to_subtype)
        match_mask = y.notna()
        X_matched = omics_data.loc[match_mask].copy()
        y_matched = y.loc[match_mask].copy()

        X_matched.index = patient_ids.loc[match_mask].values
        y_matched.index = X_matched.index
        return X_matched, y_matched

    # ------------- 预过滤 -------------
    @staticmethod
    def variance_prefilter(df: pd.DataFrame, max_features: int = None, name: str = ""):
        if (max_features is None) or (df.shape[1] <= max_features):
            return df
        variances = df.var(axis=0, numeric_only=True).fillna(0).to_numpy()
        idx = np.argsort(variances)[-max_features:]
        kept = df.iloc[:, idx]
        print(f"{name} 预过滤: {df.shape[1]} -> {kept.shape[1]}")
        return kept

    # ------------- 分批处理大矩阵 -------------
    def process_large_matrix(self, matrix, batch_size=1000, fill_method='mean'):
        n_features = matrix.shape[1]
        n_batches = (n_features + batch_size - 1) // batch_size
        processed_batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_features)
            batch = matrix.iloc[:, start_idx:end_idx].copy()
            if fill_method == 'mean':
                batch = batch.fillna(batch.mean())
            elif fill_method == 'median':
                batch = batch.fillna(batch.median())
            elif fill_method == 'zero':
                batch = batch.fillna(0)
            else:
                batch = batch.fillna(batch.mean())
            processed_batches.append(batch)
            if (i + 1) % 10 == 0:
                print(f"处理批次 {i + 1}/{n_batches}")
        return pd.concat(processed_batches, axis=1)

    # ------------- 数据增强 -------------
    @staticmethod
    def augment_data(X, y, min_samples=100):
        n_samples = len(X)
        if n_samples >= min_samples:
            return X, y
        n_needed = min_samples - n_samples
        indices = np.random.choice(n_samples, n_needed, replace=True)
        X_new = X[indices].copy()
        noise = np.random.normal(0, 0.01, X_new.shape)
        X_new = X_new + noise
        y_new = y[indices].copy()
        X_augmented = np.vstack([X, X_new])
        y_augmented = np.concatenate([y, y_new])
        print(f"数据增强: 从 {n_samples} 个样本增加到 {len(X_augmented)} 个样本")
        return X_augmented, y_augmented

    # ------------- 主准备函数 -------------
    def preprocess_data(self, clinical_file, methylation_file, cnv_file, expression_file, mutation_file=None,
                        prefilter_dna=None, prefilter_rna=None, min_samples=100):
        print("加载临床数据...")
        clinical_data = self.load_clinical_data(clinical_file)
        if len(clinical_data) == 0:
            raise ValueError("临床数据为空")

        print("加载甲基化数据...")
        meth_data = self.load_omics_data(methylation_file, "methylation")
        print(f"甲基化数据: {meth_data.shape}")

        print("加载CNV数据...")
        cnv_data = self.load_omics_data(cnv_file, "CNV")
        print(f"CNV数据: {cnv_data.shape}")

        print("加载基因表达数据...")
        expr_data = self.load_omics_data(expression_file, "expression")
        print(f"基因表达数据: {expr_data.shape}")

        mut_data = None
        if mutation_file:
            try:
                print("加载突变数据...")
                mut_data = self.load_mutation_data(mutation_file)
                print(f"突变数据: {mut_data.shape}")
            except Exception as e:
                print(f"加载突变数据失败（将忽略突变模态）: {e}")
                mut_data = None

        print("匹配样本...")
        meth_matched, y_meth = self.match_samples(meth_data, clinical_data)
        cnv_matched, y_cnv = self.match_samples(cnv_data, clinical_data)
        expr_matched, y_expr = self.match_samples(expr_data, clinical_data)

        common_samples = set(meth_matched.index) & set(cnv_matched.index) & set(expr_matched.index)

        mut_matched = None
        if mut_data is not None:
            mut_matched, y_mut = self.match_samples(mut_data, clinical_data)
            if len(mut_matched) > 0:
                common_samples = common_samples & set(mut_matched.index)

        common_samples = list(common_samples)
        print(f"共同样本数: {len(common_samples)}")
        if len(common_samples) == 0:
            raise ValueError("没有找到共同的样本")

        print("分批处理DNA数据...")
        meth_final = self.process_large_matrix(meth_matched.loc[common_samples])
        cnv_final = self.process_large_matrix(cnv_matched.loc[common_samples])
        print("分批处理RNA数据...")
        expr_final = self.process_large_matrix(expr_matched.loc[common_samples])
        y_final = y_meth.loc[common_samples]

        dna_components = [cnv_final, meth_final]
        if (mut_matched is not None) and (len(mut_matched) > 0):
            mut_final = self.process_large_matrix(mut_matched.loc[common_samples], fill_method='zero')
            dna_components.append(mut_final)
        dna_data = pd.concat(dna_components, axis=1)

        if prefilter_dna is not None:
            dna_data = self.variance_prefilter(dna_data, prefilter_dna, name="DNA")
        if prefilter_rna is not None:
            expr_final = self.variance_prefilter(expr_final, prefilter_rna, name="RNA")

        print("最终数据形状:")
        print(f"DNA数据: {dna_data.shape}")
        print(f"RNA数据: {expr_final.shape}")
        print(f"标签分布: {y_final.value_counts().to_dict()}")

        X_dna_aug, y_aug = self.augment_data(dna_data.values, y_final.values, min_samples)
        X_rna_aug, _ = self.augment_data(expr_final.values, y_final.values, min_samples)

        y_encoded = self.label_encoder.fit_transform(y_aug)

        X_dna_train, X_dna_test, X_rna_train, X_rna_test, y_train, y_test = train_test_split(
            X_dna_aug, X_rna_aug, y_encoded,
            test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_dna_train, X_dna_val, X_rna_train, X_rna_val, y_train, y_val = train_test_split(
            X_dna_train, X_rna_train, y_train,
            test_size=0.2, random_state=42, stratify=y_train
        )

        self.dna_scaler.fit(X_dna_train)
        self.rna_scaler.fit(X_rna_train)

        X_dna_train = self.dna_scaler.transform(X_dna_train)
        X_dna_val = self.dna_scaler.transform(X_dna_val)
        X_dna_test = self.dna_scaler.transform(X_dna_test)

        X_rna_train = self.rna_scaler.transform(X_rna_train)
        X_rna_val = self.rna_scaler.transform(X_rna_val)
        X_rna_test = self.rna_scaler.transform(X_rna_test)

        print("保存预处理数据...")
        sparse.save_npz(os.path.join(self.output_dir, 'dna_train.npz'), sparse.csr_matrix(X_dna_train))
        sparse.save_npz(os.path.join(self.output_dir, 'dna_val.npz'), sparse.csr_matrix(X_dna_val))
        sparse.save_npz(os.path.join(self.output_dir, 'dna_test.npz'), sparse.csr_matrix(X_dna_test))

        sparse.save_npz(os.path.join(self.output_dir, 'rna_train.npz'), sparse.csr_matrix(X_rna_train))
        sparse.save_npz(os.path.join(self.output_dir, 'rna_val.npz'), sparse.csr_matrix(X_rna_val))
        sparse.save_npz(os.path.join(self.output_dir, 'rna_test.npz'), sparse.csr_matrix(X_rna_test))

        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(self.output_dir, 'y_test.npy'), y_test)

        joblib.dump(self.label_encoder, os.path.join(self.output_dir, 'label_encoder.pkl'))
        joblib.dump(self.dna_scaler, os.path.join(self.output_dir, 'dna_scaler.pkl'))
        joblib.dump(self.rna_scaler, os.path.join(self.output_dir, 'rna_scaler.pkl'))

        print(f"预处理完成，数据已保存到: {self.output_dir}")

        return {
            'dna_train': X_dna_train, 'dna_val': X_dna_val, 'dna_test': X_dna_test,
            'rna_train': X_rna_train, 'rna_val': X_rna_val, 'rna_test': X_rna_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'label_encoder': self.label_encoder,
            'dna_scaler': self.dna_scaler,
            'rna_scaler': self.rna_scaler
        }


def main():
    """数据预处理主函数示例"""
    output_dir = "./preprocessed"

    data_files = {
        'clinical': './data/TCGA.GBM.sampleMap_GBM_clinicalMatrix',
        'methylation': './data/HumanMethylation450',
        'cnv': './data/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes',
        'expression': './data/HiSeqV2',
        'mutation': './data/GBM_mc3_gene_level.txt'  # 可选
    }

    preprocessor = MultiOmicsDataPreprocessor(data_dir="./data", output_dir=output_dir)
    preprocessor.preprocess_data(
        clinical_file=data_files['clinical'],
        methylation_file=data_files['methylation'],
        cnv_file=data_files['cnv'],
        expression_file=data_files['expression'],
        mutation_file=data_files.get('mutation', None),
        prefilter_dna=5000,
        prefilter_rna=5000,
        min_samples=100
    )


if __name__ == "__main__":
    main()
