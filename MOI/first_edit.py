import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# ============== FSD特征选择模块 ==============
class FSDSelector:
    """Feature Selection with Distribution (FSD) 模块"""
    def __init__(self, k_threshold=0.05, j_threshold=0.8, m_iterations=10):
        self.k_threshold = k_threshold  # 统计显著性阈值
        self.j_threshold = j_threshold  # 特征保留阈值
        self.m_iterations = m_iterations  # 重复次数
        self.selected_features_ = None

    def _ks_test_conditions(self, X_sub, X_full, y_sub, y_full):
        """执行三个KS检验条件"""
        try:
            _, p1 = ks_2samp(X_sub.flatten(), X_full.flatten())
        except Exception:
            p1 = 0.0
        try:
            classes = np.unique(y_full)
            if len(classes) >= 2:
                class1_data = X_full[y_full == classes[0]]
                class2_data = X_full[y_full == classes[1]]
                _, p2 = ks_2samp(class1_data.flatten(), class2_data.flatten())
            else:
                p2 = 1.0
        except Exception:
            p2 = 1.0
        try:
            classes_sub = np.unique(y_sub)
            if len(classes_sub) >= 2:
                class1_sub = X_sub[y_sub == classes_sub[0]]
                class2_sub = X_sub[y_sub == classes_sub[1]]
                _, p3 = ks_2samp(class1_sub.flatten(), class2_sub.flatten())
            else:
                p3 = 1.0
        except Exception:
            p3 = 1.0
        return p1, p2, p3

    def fit(self, X, y):
        """训练FSD选择器"""
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        feature_scores = np.zeros(n_features)

        print(f"开始FSD特征选择，总特征数: {n_features}")

        for iteration in range(self.m_iterations):
            # 随机采样子集
            n_samples = len(X)
            subset_size = max(int(0.7 * n_samples), 10)
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)

            X_sub = X[subset_indices]
            y_sub = y[subset_indices]

            # 对每个特征进行KS检验
            for feature_idx in range(n_features):
                X_feature_full = X[:, feature_idx]
                X_feature_sub = X_sub[:, feature_idx]

                # 跳过无效特征
                if np.all(np.isnan(X_feature_full)) or np.nanvar(X_feature_full) == 0:
                    continue

                p1, p2, p3 = self._ks_test_conditions(
                    X_feature_sub, X_feature_full, y_sub, y
                )

                # FSD条件: p1 > k, p2 < k, p3 < k
                if (p1 > self.k_threshold and p2 < self.k_threshold and p3 < self.k_threshold):
                    feature_scores[feature_idx] += 1

            if (iteration + 1) % 5 == 0:
                print(f"完成迭代 {iteration + 1}/{self.m_iterations}")

        # 选择满足阈值的特征
        feature_ratios = feature_scores / self.m_iterations
        self.selected_features_ = np.where(feature_ratios > self.j_threshold)[0]

        if len(self.selected_features_) == 0:
            print("警告：没有特征满足FSD条件，选择前100个特征")
            self.selected_features_ = np.argsort(feature_ratios)[-100:]

        print(f"FSD选择了 {len(self.selected_features_)} 个特征 (原始: {n_features})")
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("请先调用fit方法")
        return np.array(X)[:, self.selected_features_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

# ============== 注意力层 & 模型 ==============
class AttentionLayer(nn.Module):
    """自注意力层（返回加权和的向量；当seq_len>1时会真正做注意力聚合）"""
    def __init__(self, input_dim, hidden_dim=64, return_sequence=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequence = return_sequence

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key   = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) 或 (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        Q = self.query(x)                 # (batch, seq_len, hidden)
        K = self.key(x)                   # (batch, seq_len, hidden)
        V = self.value(x)                 # (batch, seq_len, hidden)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, seq, seq)
        attended = torch.matmul(attention_weights, V)            # (batch, seq, hidden)

        if self.return_sequence:
            return attended
        else:
            # 加权和池化到一个向量
            return attended.sum(dim=1)    # (batch, hidden)

class AttentionMOI(nn.Module):
    """Attention Multi-Omics Integration 模型"""
    def __init__(self, dna_features, rna_features, num_classes, hidden_dim=128):
        super().__init__()
        self.dna_attention = AttentionLayer(dna_features, hidden_dim, return_sequence=False)
        self.dna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.rna_attention = AttentionLayer(rna_features, hidden_dim, return_sequence=False)
        self.rna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # 跨模态注意力：输入为 (batch, 2, hidden_dim//2)
        self.cross_attention = AttentionLayer(hidden_dim // 2, hidden_dim // 2, return_sequence=False)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, dna_data, rna_data):
        dna_vec = self.dna_attention(dna_data)   # (batch, hidden)
        dna_rd  = self.dna_mlp(dna_vec)          # (batch, hidden//2)

        rna_vec = self.rna_attention(rna_data)
        rna_rd  = self.rna_mlp(rna_vec)

        # 堆成序列长度2： [DNA, RNA]
        seq = torch.stack([dna_rd, rna_rd], dim=1)  # (batch, 2, hidden//2)
        fused = self.cross_attention(seq)           # (batch, hidden//2)

        out = self.classifier(fused)
        return out

# ============== 数据加载器 ==============
class MultiOmicsDataset(Dataset):
    """多组学数据集"""
    def __init__(self, dna_data, rna_data, labels):
        self.dna_data = torch.FloatTensor(dna_data)
        self.rna_data = torch.FloatTensor(rna_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dna_data[idx], self.rna_data[idx], self.labels[idx]

# ============== 数据准备 ==============
class MultiOmicsDataLoader:
    """多组学数据加载和预处理"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.dna_scaler = StandardScaler()
        self.rna_scaler = StandardScaler()
        # 完整条形码: TCGA-XX-XXXX-YY[Letter]...
        self.full_re = re.compile(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})[A-Z]?.*')

    # ------------- 条形码辅助 -------------
    def _extract_patient_id(self, s: str):
        if not isinstance(s, str):
            return None
        m = self.full_re.match(s.strip())
        return m.group(1) if m else None  # 病人级短ID: TCGA-XX-XXXX

    def _extract_sample_code(self, s: str):
        if not isinstance(s, str):
            return None
        m = self.full_re.match(s.strip())
        return m.group(2) if m else None   # 两位样本类型编码

    def _is_primary_by_barcode(self, s: str) -> bool:
        code = self._extract_sample_code(s)
        return True if code is None else (code == "01")  # 临床若只有病人级ID则视为有效

    # ------------- 读文件 -------------
    def _read_tsv(self, filepath):
        try:
            return pd.read_csv(filepath, sep='\t', low_memory=False)
        except Exception:
            return pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)

    # ------------- 临床数据 -------------
    def load_clinical_data(self, clinical_file):
        df = self._read_tsv(clinical_file)

        # A：属性在行，样本在列
        tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
        if len(tcga_cols) >= 5:  # 大概率宽表
            attr_col = df.columns[0]
            row_mask = df[attr_col].astype(str).str.contains(
                r'(subtype|GeneExp_Subtype|Subtype_mRNA|mRNA.*cluster|verhaak)',
                case=False, regex=True, na=False
            )
            if not row_mask.any():
                raise ValueError("在临床矩阵里找不到亚型行（包含 'subtype' 或 'GeneExp_Subtype' 等字样）")
            row = df.loc[row_mask].iloc[0]
            sub_series = row.drop(labels=[attr_col])
            clin = pd.DataFrame({
                "sample_barcode": sub_series.index.astype(str),
                "subtype": sub_series.values
            })
        else:
            # 情形B：样本在行
            sid_col = None
            for col in ["sampleID", "bcr_sample_barcode", "sample", "barcode", "Sample", "SAMPLE_ID"]:
                if col in df.columns:
                    sid_col = col; break
            if sid_col is None:
                cand = [c for c in df.columns if df[c].astype(str).str.startswith("TCGA-").any()]
                sid_col = cand[0] if cand else df.columns[0]

            subtype_col = None
            for col in ["GeneExp_Subtype", "Subtype_mRNA", "Subtype", "subtype", "mRNA_cluster"]:
                if col in df.columns:
                    subtype_col = col; break
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
        clin["is_primary"]  = clin["sample_barcode"].apply(self._is_primary_by_barcode)
        clin["subtype"]     = clin["subtype"].apply(norm_subtype)

        clin = clin[clin["patient_id"].notna() & clin["subtype"].notna()]
        clin = (clin.sort_values(["patient_id", "is_primary"], ascending=[True, False])
                    .drop_duplicates(subset=["patient_id"], keep="first"))

        out = clin[["patient_id", "subtype"]].rename(columns={"patient_id": "short_id"})
        print(f"临床数据（去重后病人数）: {len(out)}")
        return out

    # ------------- 组学数据 -------------
    def load_omics_data(self, omics_file, data_type="methylation"):
        df = self._read_tsv(omics_file)

        # 如果第一列是特征名（基因/探针），则设为索引
        first_col_lower = str(df.columns[0]).strip().lower()
        if first_col_lower in {"gene","genes","hugo","symbol","gene symbol","id","probe","probes","composite element ref"}:
            df = df.set_index(df.columns[0])

        # 判断样本是否在列
        def cols_have_tcga(_df):
            return any(str(c).startswith("TCGA-") for c in _df.columns)

        if not cols_have_tcga(df):
            # 检查第一列是否为样本ID
            first_col = df.columns[0]
            if df[first_col].astype(str).str.startswith("TCGA-").any():
                df = df.set_index(first_col).transpose()

        # 如果还没有样本在列，看看index是不是样本，再转置
        if not any(str(c).startswith("TCGA-") for c in df.columns):
            if any(str(i).startswith("TCGA-") for i in df.index):
                df = df.transpose()

        # 现在列应是样本
        tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
        if len(tcga_cols) == 0:
            raise ValueError(f"{data_type}数据中未发现TCGA样本")

        df = df[tcga_cols]
        df = df.apply(pd.to_numeric, errors="coerce")

        X = df.transpose().copy()  # (样本, 特征)
        X.index.name = "sample_barcode"
        return X

    def load_mutation_data(self, mutation_file):
        return self.load_omics_data(mutation_file, "mutation")

    # ------------- 样本匹配（病人级） -------------
    def match_samples(self, omics_data: pd.DataFrame, clinical_data: pd.DataFrame):
        # 将样本条形码映射为病人级短ID
        patient_ids = omics_data.index.to_series().astype(str).apply(self._extract_patient_id)
        keep = patient_ids.notna()
        omics_data = omics_data.loc[keep].copy()
        patient_ids = patient_ids.loc[keep]

        # 去重（同一病人可能多个测次），保留首次
        dedup_mask = ~patient_ids.duplicated()
        omics_data = omics_data.loc[dedup_mask]
        patient_ids = patient_ids.loc[dedup_mask]

        short_to_subtype = dict(zip(clinical_data["short_id"], clinical_data["subtype"]))
        y = patient_ids.map(short_to_subtype)

        match_mask = y.notna()
        X_matched = omics_data.loc[match_mask].copy()
        y_matched = y.loc[match_mask].copy()

        # 索引改为病人级短ID，便于后续多模态对齐
        X_matched.index = patient_ids.loc[match_mask].values
        y_matched.index = X_matched.index

        return X_matched, y_matched

    # ------------- 预过滤（可选：控制维度，避免极大特征数拖慢FSD） -------------
    @staticmethod
    def variance_prefilter(df: pd.DataFrame, max_features: int = None, name: str = ""):
        if (max_features is None) or (df.shape[1] <= max_features):
            return df
        variances = df.var(axis=0, numeric_only=True).fillna(0).to_numpy()
        idx = np.argsort(variances)[-max_features:]
        kept = df.iloc[:, idx]
        print(f"{name} 预过滤: {df.shape[1]} -> {kept.shape[1]}（按方差保留前 {max_features} 个特征）")
        return kept

    # ------------- 主准备函数 -------------
    def prepare_data(self, clinical_file, methylation_file, cnv_file, expression_file, mutation_file=None,
                     prefilter_dna=None, prefilter_rna=None):
        print("加载临床数据...")
        clinical_data = self.load_clinical_data(clinical_file)
        if len(clinical_data) == 0:
            raise ValueError("临床数据为空：请检查clinicalMatrix/窄表中的亚型行或列命名是否正确。")

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

        # 匹配到临床（病人级）
        print("匹配样本...")
        meth_matched, y_meth = self.match_samples(meth_data, clinical_data)
        cnv_matched,  y_cnv  = self.match_samples(cnv_data,  clinical_data)
        expr_matched, y_expr = self.match_samples(expr_data, clinical_data)

        common_samples = set(meth_matched.index) & set(cnv_matched.index) & set(expr_matched.index)

        mut_matched = None
        if mut_data is not None:
            mut_matched, y_mut = self.match_samples(mut_data, clinical_data)
            print(f"突变数据匹配样本数: {len(mut_matched)}")
            if len(mut_matched) > 0:
                common_samples = common_samples & set(mut_matched.index)

        common_samples = list(common_samples)
        print(f"共同样本数: {len(common_samples)}")
        if len(common_samples) == 0:
            raise ValueError("没有找到共同的样本 —— 先运行 quick_inspect_paths(...) 核对各文件的样本ID与交集。")

        # 对齐 & 填补缺失
        meth_final = meth_matched.loc[common_samples].fillna(meth_matched.mean())
        cnv_final  = cnv_matched.loc[common_samples].fillna(cnv_matched.mean())
        expr_final = expr_matched.loc[common_samples].fillna(expr_matched.mean())
        y_final    = y_meth.loc[common_samples]

        dna_components = [cnv_final, meth_final]
        if (mut_matched is not None) and (len(mut_matched) > 0):
            mut_final = mut_matched.loc[common_samples].fillna(0)  # 突变二值用0填
            dna_components.append(mut_final)
            print(f"DNA数据包含: CNV({cnv_final.shape[1]}) + 甲基化({meth_final.shape[1]}) + 突变({mut_final.shape[1]})")
        else:
            print(f"DNA数据包含: CNV({cnv_final.shape[1]}) + 甲基化({meth_final.shape[1]})")

        dna_data = pd.concat(dna_components, axis=1)

        # 可选：预过滤降维（避免FSD在超高维上过慢）
        if prefilter_dna is not None:
            dna_data = self.variance_prefilter(dna_data, prefilter_dna, name="DNA")
        if prefilter_rna is not None:
            expr_final = self.variance_prefilter(expr_final, prefilter_rna, name="RNA")

        print("最终数据形状:")
        print(f"DNA数据: {dna_data.shape}")
        print(f"RNA数据: {expr_final.shape}")
        print(f"标签分布: {y_final.value_counts().to_dict()}")

        return dna_data, expr_final, y_final

# ============== 训练器 ==============
class AttentionMOITrainer:
    """AttentionMOI训练器"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_model_state = None

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for dna_data, rna_data, labels in dataloader:
            dna_data = dna_data.to(self.device)
            rna_data = rna_data.to(self.device)
            labels   = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(dna_data, rna_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for dna_data, rna_data, labels in dataloader:
                dna_data = dna_data.to(self.device)
                rna_data = rna_data.to(self.device)
                labels   = labels.to(self.device)

                outputs = self.model(dna_data, rna_data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=300, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

        best_val_acc = 0.0
        patience = 30
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"早停于epoch {epoch+1}")
                break

            scheduler.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.4f}  '
                      f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}')

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
        return best_val_acc

# ============== 输入检查器 ==============
def quick_inspect_paths(clinical_file, methylation_file, cnv_file, expression_file, mutation_file=None):
    print("="*60, "\n[1] 检查临床文件")
    try:
        df = pd.read_csv(clinical_file, sep="\t", low_memory=False)
    except Exception:
        df = pd.read_csv(clinical_file, sep="\t", encoding="latin-1", low_memory=False)

    tcga_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
    if len(tcga_cols) >= 5:
        print(f"临床是宽表（属性在行，样本在列），列数={len(df.columns)}，样本列≈{len(tcga_cols)}")
        print("样本列例子：", tcga_cols[:5])
        attrs = df.iloc[:10,0].tolist()
        print("前10个属性名示例：", attrs)
    else:
        print(f"临床是窄表（样本在行），列名示例：{list(df.columns[:8])}")
        sid_col = None
        for col in ["sampleID","bcr_sample_barcode","sample","barcode","Sample","SAMPLE_ID"]:
            if col in df.columns:
                sid_col = col; break
        if sid_col is None:
            cand = [c for c in df.columns if df[c].astype(str).str.startswith("TCGA-").any()]
            sid_col = cand[0] if cand else df.columns[0]
        print("推测样本列：", sid_col)
        print("样本ID示例：", df[sid_col].dropna().astype(str).head(5).tolist())

    loader = MultiOmicsDataLoader('data/')
    print("\n[2] 检查各组学矩阵（转成样本×特征后）")
    for pth, kind in [(methylation_file,"methylation"),
                      (cnv_file,"cnv"),
                      (expression_file,"expression"),
                      (mutation_file,"mutation")]:
        if pth is None:
            continue
        try:
            X = loader.load_omics_data(pth, kind if kind!="mutation" else "mutation")
            print(f"{kind}: 形状={X.shape}")
            idx_list = list(map(str, X.index[:5]))
            print(f"{kind}: 样本条形码示例：", idx_list)
            short_ids = [loader._extract_patient_id(s) for s in idx_list]
            print(f"{kind}: 病人级ID示例   ：", short_ids)
        except Exception as e:
            print(f"{kind}: 解析失败 -> {e}")

    print("\n[3] 交集规模（病人级短ID）")
    try:
        clin_ok = loader.load_clinical_data(clinical_file)
        print("临床病人数：", len(clin_ok))
        short_clin = set(clin_ok["short_id"])
        mats = {}
        for pth, kind in [(methylation_file,"methylation"),
                          (cnv_file,"cnv"),
                          (expression_file,"expression"),
                          (mutation_file,"mutation")]:
            if pth is None:
                continue
            X = loader.load_omics_data(pth, kind if kind!="mutation" else "mutation")
            shorts = set(X.index.to_series().astype(str).apply(loader._extract_patient_id).dropna().unique())
            mats[kind] = shorts
            print(f"{kind}: 样本病人数={len(shorts)}, 与临床交集={len(shorts & short_clin)}")
        if mats:
            inter = short_clin.copy()
            for s in mats.values():
                inter &= s
            print("临床∩全部组学 的病人交集规模：", len(inter))
            if len(inter) > 0:
                print("交集里的前几个病人ID：", list(inter)[:5])
    except Exception as e:
        print("交集统计失败：", e)

# ============== 主执行函数 ==============
def main():
    """主函数：演示完整的AttentionMOI流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data_files = {
        'clinical':   './data/TCGA.GBM.sampleMap_GBM_clinicalMatrix',
        'methylation':'./data/HumanMethylation450',
        'cnv':        './data/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes',
        'expression': './data/HiSeqV2',
        'mutation':   './data/GBM_mc3_gene_level.txt'
    }

    try:
        print("=" * 50)
        print("步骤0: 快速检查输入")
        print("=" * 50)
        quick_inspect_paths(
            data_files['clinical'],
            data_files['methylation'],
            data_files['cnv'],
            data_files['expression'],
            data_files['mutation']
        )

        print("\n" + "=" * 50)
        print("步骤1: 数据加载与预处理")
        print("=" * 50)
        data_loader = MultiOmicsDataLoader('data/')

        # 可选预过滤：防止超高维拖慢FSD（根据你的机器/数据量调整阈值或设为None）
        dna_prefilter_max = 50000   # DNA（CNV+甲基化+突变）最多保留多少特征
        rna_prefilter_max = 15000   # RNA最多保留多少特征

        dna_data, rna_data, labels = data_loader.prepare_data(
            data_files['clinical'],
            data_files['methylation'],
            data_files['cnv'],
            data_files['expression'],
            data_files['mutation'],
            prefilter_dna=dna_prefilter_max,
            prefilter_rna=rna_prefilter_max
        )

        print("\n" + "=" * 50)
        print("步骤2: FSD特征选择")
        print("=" * 50)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)

        print("DNA特征选择...")
        fsd_dna = FSDSelector(k_threshold=0.05, j_threshold=0.3, m_iterations=3)
        dna_selected = fsd_dna.fit_transform(dna_data.values, y_encoded)

        print("RNA特征选择...")
        fsd_rna = FSDSelector(k_threshold=0.05, j_threshold=0.3, m_iterations=3)
        rna_selected = fsd_rna.fit_transform(rna_data.values, y_encoded)

        print(f"DNA特征: {dna_data.shape[1]} -> {dna_selected.shape[1]}")
        print(f"RNA特征: {rna_data.shape[1]} -> {rna_selected.shape[1]}")

        print("\n" + "=" * 50)
        print("步骤3: 数据标准化")
        print("=" * 50)
        scaler_dna = StandardScaler()
        scaler_rna = StandardScaler()
        dna_scaled = scaler_dna.fit_transform(dna_selected)
        rna_scaled = scaler_rna.fit_transform(rna_selected)

        print("数据分割...")
        X_dna_train, X_dna_test, X_rna_train, X_rna_test, y_train, y_test = train_test_split(
            dna_scaled, rna_scaled, y_encoded,
            test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_dna_train, X_dna_val, X_rna_train, X_rna_val, y_train, y_val = train_test_split(
            X_dna_train, X_rna_train, y_train,
            test_size=0.2, random_state=42, stratify=y_train
        )

        print("创建数据加载器...")
        train_dataset = MultiOmicsDataset(X_dna_train, X_rna_train, y_train)
        val_dataset   = MultiOmicsDataset(X_dna_val,   X_rna_val,   y_val)
        test_dataset  = MultiOmicsDataset(X_dna_test,  X_rna_test,  y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

        print("\n" + "=" * 50)
        print("步骤4: 创建AttentionMOI模型")
        print("=" * 50)
        num_classes = len(np.unique(y_encoded))
        model = AttentionMOI(
            dna_features=dna_selected.shape[1],
            rna_features=rna_selected.shape[1],
            num_classes=num_classes,
            hidden_dim=128
        )

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"类别数量: {num_classes}")
        print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")

        print("\n" + "=" * 50)
        print("步骤5: 模型训练")
        print("=" * 50)
        trainer = AttentionMOITrainer(model, device)
        best_val_acc = trainer.train(train_loader, val_loader, epochs=100)

        print("\n" + "=" * 50)
        print("步骤6: 模型评估")
        print("=" * 50)
        test_loss, test_acc, test_preds, test_labels = trainer.evaluate(test_loader, nn.CrossEntropyLoss())

        precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        recall    = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
        f1        = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

        print("测试结果:")
        print(f"准确率: {test_acc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")

        print(f"\n类别映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        from collections import Counter
        print(f"真实标签分布: {Counter(test_labels)}")
        print(f"预测标签分布: {Counter(test_preds)}")

        print("\n" + "=" * 50)
        print("AttentionMOI多组学整合完成！")
        print("=" * 50)

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
