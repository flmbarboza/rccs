import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class CreditScoringAI:
    def __init__(self):
        self.model_suggestions = {
            "logistic_regression": "Regressão Logística - Modelo interpretável, ideal para explicar decisões de crédito",
            "random_forest": "Random Forest - Modelo robusto que captura interações complexas entre variáveis",
            "gradient_boosting": "Gradient Boosting - Excelente performance, mas requer mais cuidado com overfitting"
        }
    
    def analyze_data_quality(self, df):
        """Análise completa da qualidade dos dados com sugestões da IA"""
        quality_report = {
            'problemas_detectados': [],
            'sugestoes_nomes': {},
            'sugestoes_valores': {},
            'sugestoes_fusao': [],
            'sugestoes_preprocessamento': []
        }
        
        for col in df.columns:
            # Análise de nomes das colunas
            if ' ' in col or col.isupper() or any(char in col for char in ['/', '(', ')', '-']):
                new_name = col.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')
                quality_report['sugestoes_nomes'][col] = {
                    'novo_nome': new_name,
                    'motivo': 'Padronização: minúsculas, sem espaços ou caracteres especiais'
                }
            
            # Análise de valores missing
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 5:
                quality_report['problemas_detectados'].append({
                    'coluna': col,
                    'problema': f'{missing_pct:.1f}% de dados faltantes',
                    'severidade': 'Alta' if missing_pct > 20 else 'Média',
                    'sugestao': self._suggest_missing_treatment(df, col, missing_pct)
                })
            
            # Análise de inconsistências
            if df[col].dtype == 'object':
                # Detectar possíveis inconsistências em categóricas
                value_counts = df[col].value_counts()
                similar_values = self._detect_similar_values(value_counts.index.tolist())
                if similar_values:
                    quality_report['sugestoes_valores'][col] = similar_values
            
            elif df[col].dtype in ['int64', 'float64']:
                # Detectar outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                
                if len(outliers) > len(df) * 0.05:  # Mais de 5% outliers
                    quality_report['problemas_detectados'].append({
                        'coluna': col,
                        'problema': f'{len(outliers)} outliers detectados ({len(outliers)/len(df)*100:.1f}%)',
                        'severidade': 'Média',
                        'sugestao': f'Considere winsorização ou transformação log para {col}'
                    })
        
        # Sugestões de fusão de variáveis
        quality_report['sugestoes_fusao'] = self._suggest_variable_fusion(df)
        
        # Sugestões de pré-processamento
        quality_report['sugestoes_preprocessamento'] = self._suggest_preprocessing(df)
        
        return quality_report
    
    def _suggest_missing_treatment(self, df, col, missing_pct):
        """Sugere tratamento para dados faltantes baseado no tipo e padrão"""
        if df[col].dtype in ['int64', 'float64']:
            if missing_pct > 30:
                return f"Alto percentual missing. Considere: criar flag 'is_{col}_missing' e imputar com mediana"
            else:
                return f"Imputar com mediana ou usar algoritmos que lidam com missing (Random Forest)"
        else:
            if missing_pct > 30:
                return f"Criar categoria 'Não_Informado' ou remover variável se não for crítica"
            else:
                return f"Imputar com moda ou criar categoria 'Outros'"
    
    def _detect_similar_values(self, values):
        """Detecta valores similares que podem ser padronizados"""
        suggestions = {}
        
        # Detectar variações de caso
        values_lower = [str(v).lower() for v in values if pd.notna(v)]
        value_groups = {}
        
        for val in values:
            if pd.isna(val):
                continue
            val_lower = str(val).lower()
            if val_lower not in value_groups:
                value_groups[val_lower] = []
            value_groups[val_lower].append(val)
        
        # Encontrar grupos com múltiplas variações
        for key, group in value_groups.items():
            if len(group) > 1:
                suggestions[f"Grupo '{key}'"] = {
                    'valores_originais': group,
                    'valor_sugerido': max(group, key=len),  # Valor mais longo como padrão
                    'motivo': 'Variações de capitalização detectadas'
                }
        
        return suggestions
    
    def _suggest_variable_fusion(self, df):
        """Sugere possíveis fusões de variáveis"""
        fusion_suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Sugerir fusões baseadas em correlação alta
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
            
            for col1, col2, corr in high_corr_pairs:
                fusion_suggestions.append({
                    'tipo': 'fusao_alta_correlacao',
                    'variaveis': [col1, col2],
                    'correlacao': corr,
                    'sugestao': f'Criar índice combinado: ({col1} + {col2})/2 ou usar PCA',
                    'motivo': f'Correlação muito alta ({corr:.3f}) pode causar multicolinearidade'
                })
        
        # Sugerir criação de variáveis derivadas
        if 'idade' in df.columns:
            fusion_suggestions.append({
                'tipo': 'variavel_derivada',
                'variaveis': ['idade'],
                'sugestao': 'Criar faixas etárias: Jovem(18-30), Adulto(31-50), Sênior(51+)',
                'motivo': 'Relação não-linear com risco de crédito'
            })
        
        if 'renda' in df.columns:
            fusion_suggestions.append({
                'tipo': 'variavel_derivada',
                'variaveis': ['renda'],
                'sugestao': 'Criar classes de renda ou aplicar transformação log',
                'motivo': 'Distribuição geralmente assimétrica'
            })
        
        return fusion_suggestions
    
    def _suggest_preprocessing(self, df):
        """Sugestões gerais de pré-processamento"""
        suggestions = []
        
        # Encoding de categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            suggestions.append({
                'etapa': 'encoding_categoricas',
                'colunas': list(categorical_cols),
                'sugestao': 'Aplicar One-Hot Encoding ou Target Encoding',
                'detalhes': 'One-Hot para poucas categorias, Target Encoding para muitas'
            })
        
        # Normalização
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            suggestions.append({
                'etapa': 'normalizacao',
                'colunas': list(numeric_cols),
                'sugestao': 'Aplicar StandardScaler ou MinMaxScaler',
                'detalhes': 'Especialmente importante para Regressão Logística'
            })
        
        # Feature Selection
        if len(df.columns) > 20:
            suggestions.append({
                'etapa': 'selecao_features',
                'colunas': list(df.columns),
                'sugestao': 'Aplicar seleção de features',
                'detalhes': 'Use RFE, SelectKBest ou importância por árvores'
            })
        
        return suggestions
    
    def apply_data_corrections(self, df, corrections_config):
        """Aplica correções baseadas na configuração do usuário"""
        df_corrected = df.copy()
        
        # Renomear colunas
        if 'rename_columns' in corrections_config:
            df_corrected = df_corrected.rename(columns=corrections_config['rename_columns'])
        
        # Tratar valores faltantes
        if 'missing_treatment' in corrections_config:
            for col, method in corrections_config['missing_treatment'].items():
                if method == 'median':
                    df_corrected[col].fillna(df_corrected[col].median(), inplace=True)
                elif method == 'mode':
                    df_corrected[col].fillna(df_corrected[col].mode()[0], inplace=True)
                elif method == 'flag_and_median':
                    df_corrected[f'{col}_missing'] = df_corrected[col].isnull().astype(int)
                    df_corrected[col].fillna(df_corrected[col].median(), inplace=True)
                elif method == 'category_missing':
                    df_corrected[col].fillna('Não_Informado', inplace=True)
        
        # Padronizar valores categóricos
        if 'standardize_values' in corrections_config:
            for col, mappings in corrections_config['standardize_values'].items():
                df_corrected[col] = df_corrected[col].replace(mappings)
        
        # Tratar outliers
        if 'outlier_treatment' in corrections_config:
            for col, method in corrections_config['outlier_treatment'].items():
                if method == 'winsorize':
                    Q1 = df_corrected[col].quantile(0.05)
                    Q3 = df_corrected[col].quantile(0.95)
                    df_corrected[col] = df_corrected[col].clip(lower=Q1, upper=Q3)
                elif method == 'log_transform':
                    df_corrected[col] = np.log1p(df_corrected[col])
        
        # Criar variáveis derivadas
        if 'create_derived' in corrections_config:
            for config in corrections_config['create_derived']:
                if config['type'] == 'age_groups':
                    col = config['column']
                    df_corrected[f'{col}_grupo'] = pd.cut(
                        df_corrected[col], 
                        bins=[0, 30, 50, 100], 
                        labels=['Jovem', 'Adulto', 'Senior']
                    )
                elif config['type'] == 'income_log':
                    col = config['column']
                    df_corrected[f'{col}_log'] = np.log1p(df_corrected[col])
        
        return df_corrected
        
    def analyze_univariate(self, df, column):
        """Análise univariada automatizada com sugestões da IA"""
        analysis = {}
        
        if df[column].dtype in ['int64', 'float64']:
            # Variável numérica
            analysis['tipo'] = 'numérica'
            analysis['estatisticas'] = {
                'média': df[column].mean(),
                'mediana': df[column].median(),
                'desvio_padrão': df[column].std(),
                'mín': df[column].min(),
                'máx': df[column].max(),
                'missing': df[column].isnull().sum()
            }
            
            # Sugestões da IA
            cv = analysis['estatisticas']['desvio_padrão'] / analysis['estatisticas']['média']
            if cv > 1:
                analysis['sugestao_ia'] = f"⚠️ A variável '{column}' apresenta alta variabilidade (CV={cv:.2f}). Considere transformações como log ou binning para melhorar a modelagem."
            elif analysis['estatisticas']['missing'] > len(df) * 0.1:
                analysis['sugestao_ia'] = f"⚠️ A variável '{column}' possui {analysis['estatisticas']['missing']} valores missing ({analysis['estatisticas']['missing']/len(df)*100:.1f}%). Considere estratégias de imputação ou remoção."
            else:
                analysis['sugestao_ia'] = f"✅ A variável '{column}' apresenta distribuição adequada para modelagem."
                
        else:
            # Variável categórica
            analysis['tipo'] = 'categórica'
            analysis['estatisticas'] = {
                'categorias': df[column].nunique(),
                'moda': df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A',
                'missing': df[column].isnull().sum(),
                'distribuicao': df[column].value_counts().to_dict()
            }
            
            # Sugestões da IA
            if analysis['estatisticas']['categorias'] > 10:
                analysis['sugestao_ia'] = f"⚠️ A variável '{column}' possui muitas categorias ({analysis['estatisticas']['categorias']}). Considere agrupamento das categorias menos frequentes."
            else:
                analysis['sugestao_ia'] = f"✅ A variável '{column}' possui número adequado de categorias para modelagem."
                
        return analysis
    
    def analyze_bivariate(self, df, var1, var2, target=None):
        """Análise bivariada com insights da IA"""
        analysis = {}
        
        # Correlação se ambas numéricas
        if df[var1].dtype in ['int64', 'float64'] and df[var2].dtype in ['int64', 'float64']:
            correlation = df[var1].corr(df[var2])
            analysis['correlacao'] = correlation
            
            if abs(correlation) > 0.7:
                analysis['sugestao_ia'] = f"⚠️ Alta correlação ({correlation:.3f}) entre {var1} e {var2}. Considere remover uma das variáveis para evitar multicolinearidade."
            elif abs(correlation) > 0.3:
                analysis['sugestao_ia'] = f"📊 Correlação moderada ({correlation:.3f}) entre {var1} e {var2}. Ambas podem ser úteis no modelo."
            else:
                analysis['sugestao_ia'] = f"✅ Baixa correlação ({correlation:.3f}) entre {var1} e {var2}. Variáveis independentes."
        
        # Análise com target se disponível
        if target and target in df.columns:
            if df[var1].dtype in ['int64', 'float64']:
                target_corr = df[var1].corr(df[target])
                analysis['correlacao_target'] = target_corr
                
                if abs(target_corr) > 0.3:
                    analysis['sugestao_ia'] += f"\n🎯 Excelente: {var1} tem boa correlação com o target ({target_corr:.3f}). Variável preditiva importante!"
                else:
                    analysis['sugestao_ia'] += f"\n📈 {var1} tem correlação baixa com o target ({target_corr:.3f}). Pode não ser muito preditiva."
        
        return analysis
    
    def suggest_model_strategy(self, df, target_column):
        """Sugestões estratégicas da IA para modelagem"""
        suggestions = []
        
        # Análise do dataset
        n_rows, n_cols = df.shape
        missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100
        
        # Análise do target
        if target_column in df.columns:
            target_balance = df[target_column].value_counts()
            minority_pct = target_balance.min() / target_balance.sum() * 100
            
            if minority_pct < 10:
                suggestions.append(f"⚠️ Dataset desbalanceado ({minority_pct:.1f}% classe minoritária). Considere técnicas de balanceamento como SMOTE ou ajuste de pesos.")
            
            if n_rows < 1000:
                suggestions.append("📊 Dataset pequeno. Regressão Logística pode ser mais estável que modelos complexos.")
            elif n_rows > 50000:
                suggestions.append("🚀 Dataset grande. Modelos como Random Forest ou Gradient Boosting podem capturar padrões complexos.")
        
        if missing_pct > 15:
            suggestions.append(f"🔧 Alto percentual de dados faltantes ({missing_pct:.1f}%). Implemente estratégia robusta de imputação.")
        
        if n_cols > 50:
            suggestions.append("🎯 Muitas variáveis. Considere seleção de features com técnicas como RFE ou importância por árvores.")
        
        return suggestions
    
    def generate_performance_insights(self, y_true, y_pred, y_proba=None):
        """Insights de performance do modelo"""
        insights = []
        
        # AUC Score
        if y_proba is not None:
            auc = roc_auc_score(y_true, y_proba)
            if auc > 0.8:
                insights.append(f"🎉 Excelente performance! AUC = {auc:.3f}. O modelo tem alta capacidade discriminatória.")
            elif auc > 0.7:
                insights.append(f"👍 Boa performance. AUC = {auc:.3f}. O modelo é adequado para uso em produção.")
            elif auc > 0.6:
                insights.append(f"⚠️ Performance moderada. AUC = {auc:.3f}. Considere engenharia de features ou outros algoritmos.")
            else:
                insights.append(f"❌ Performance baixa. AUC = {auc:.3f}. O modelo precisa ser reformulado.")
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Métricas de negócio
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        insights.append(f"📊 Precisão: {precision:.3f} | Recall: {recall:.3f}")
        
        # Análise de negócio
        if fp > fn:
            insights.append("💰 Modelo conservador: mais falsos positivos (rejeita bons clientes). Considere ajustar threshold para reduzir perdas de negócio.")
        elif fn > fp:
            insights.append("⚠️ Modelo arriscado: mais falsos negativos (aprova maus clientes). Considere aumentar threshold para reduzir inadimplência.")
        
        return insights
    
    def suggest_credit_policies(self, model_performance, business_context):
        """Sugestões de políticas de crédito baseadas na IA"""
        policies = []
        
        policies.append("🎯 **Políticas de Aprovação Recomendadas:**")
        
        if business_context.get('risk_appetite', 'medium') == 'low':
            policies.append("- Threshold conservador: Score > 0.7 para aprovação automática")
            policies.append("- Análise manual para scores entre 0.5-0.7")
            policies.append("- Rejeição automática para scores < 0.5")
        elif business_context.get('risk_appetite', 'medium') == 'high':
            policies.append("- Threshold agressivo: Score > 0.4 para aprovação automática")
            policies.append("- Análise manual apenas para scores < 0.3")
        else:
            policies.append("- Threshold moderado: Score > 0.6 para aprovação automática")
            policies.append("- Análise manual para scores entre 0.4-0.6")
            policies.append("- Rejeição automática para scores < 0.4")
        
        policies.append("\n📋 **Recomendações de Monitoramento:**")
        policies.append("- Revisar modelo mensalmente")
        policies.append("- Monitorar PSI (Population Stability Index)")
        policies.append("- Acompanhar taxa de aprovação e inadimplência")
        
        return policies

def main():
    st.set_page_config(page_title="Assistente IA - Credit Scoring", layout="wide")
    
    st.title("🤖 Assistente IA para Análise de Credit Scoring")
    st.markdown("*Seu assistente inteligente para análise de risco de crédito*")
    
    # Inicializar IA
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = CreditScoringAI()
    
    ai = st.session_state.ai_assistant
    
    # Sidebar para upload e configurações
    with st.sidebar:
        st.header("📊 Configurações")
        
        uploaded_file = st.file_uploader("Upload do dataset", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
            
            # Seleção da variável target
            target_col = st.selectbox("Selecione a variável target (inadimplência):", df.columns)
            
            # Contexto de negócio
            st.subheader("🏦 Contexto de Negócio")
            risk_appetite = st.selectbox("Apetite ao risco:", ["low", "medium", "high"])
            business_context = {"risk_appetite": risk_appetite}
    
    # Tabs principais
    if 'df' in locals():
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 Qualidade dos Dados", "📊 Análise Univariada", "🔗 Análise Bivariada", "🤖 Modelagem", "📈 Performance", "📋 Políticas"])
        
        with tab0:
            st.header("🔧 Análise e Correção da Qualidade dos Dados")
            
            # Análise automática da qualidade
            if st.button("🔍 Analisar Qualidade dos Dados"):
                with st.spinner("Analisando qualidade dos dados..."):
                    quality_report = ai.analyze_data_quality(df)
                    st.session_state.quality_report = quality_report
                    st.success("✅ Análise de qualidade concluída!")
            
            if 'quality_report' in st.session_state:
                report = st.session_state.quality_report
                
                # Problemas detectados
                if report['problemas_detectados']:
                    st.subheader("⚠️ Problemas Detectados")
                    for problema in report['problemas_detectados']:
                        with st.expander(f"{problema['coluna']}: {problema['problema']} - Severidade: {problema['severidade']}"):
                            st.write(f"**Sugestão da IA:** {problema['sugestao']}")
                
                # Painel interativo de correções
                st.subheader("🛠️ Painel de Correções Interativo")
                
                corrections_config = {}
                
                # Renomear colunas
                if report['sugestoes_nomes']:
                    st.write("**📝 Renomear Colunas:**")
                    rename_dict = {}
                    for old_name, suggestion in report['sugestoes_nomes'].items():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        col1.write(f"**{old_name}**")
                        new_name = col2.text_input(f"Novo nome", suggestion['novo_nome'], key=f"rename_{old_name}")
                        if col3.checkbox("Aplicar", key=f"apply_rename_{old_name}"):
                            rename_dict[old_name] = new_name
                    if rename_dict:
                        corrections_config['rename_columns'] = rename_dict
                
                # Tratar dados faltantes
                st.write("**🔍 Tratamento de Dados Faltantes:**")
                missing_treatment = {}
                for col in df.columns:
                    missing_pct = df[col].isnull().sum() / len(df) * 100
                    if missing_pct > 0:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        col1.write(f"**{col}** ({missing_pct:.1f}% missing)")
                        
                        if df[col].dtype in ['int64', 'float64']:
                            options = ['Não tratar', 'Mediana', 'Flag + Mediana']
                            method = col2.selectbox("Método", options, key=f"missing_{col}")
                            if col3.checkbox("Aplicar", key=f"apply_missing_{col}") and method != 'Não tratar':
                                if method == 'Mediana':
                                    missing_treatment[col] = 'median'
                                elif method == 'Flag + Mediana':
                                    missing_treatment[col] = 'flag_and_median'
                        else:
                            options = ['Não tratar', 'Moda', 'Categoria "Não_Informado"']
                            method = col2.selectbox("Método", options, key=f"missing_{col}")
                            if col3.checkbox("Aplicar", key=f"apply_missing_{col}") and method != 'Não tratar':
                                if method == 'Moda':
                                    missing_treatment[col] = 'mode'
                                elif method == 'Categoria "Não_Informado"':
                                    missing_treatment[col] = 'category_missing'
                
                if missing_treatment:
                    corrections_config['missing_treatment'] = missing_treatment
                
                # Padronizar valores categóricos
                if report['sugestoes_valores']:
                    st.write("**🔤 Padronização de Valores Categóricos:**")
                    standardize_values = {}
                    for col, suggestions in report['sugestoes_valores'].items():
                        st.write(f"**Coluna: {col}**")
                        col_mappings = {}
                        for group_name, group_info in suggestions.items():
                            with st.expander(f"{group_name}"):
                                st.write(f"Valores encontrados: {group_info['valores_originais']}")
                                new_value = st.text_input(f"Valor padronizado", group_info['valor_sugerido'], key=f"std_{col}_{group_name}")
                                if st.checkbox(f"Aplicar padronização", key=f"apply_std_{col}_{group_name}"):
                                    for orig_val in group_info['valores_originais']:
                                        col_mappings[orig_val] = new_value
                        if col_mappings:
                            standardize_values[col] = col_mappings
                    
                    if standardize_values:
                        corrections_config['standardize_values'] = standardize_values
                
                # Tratamento de outliers
                st.write("**📊 Tratamento de Outliers:**")
                outlier_treatment = {}
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                    
                    if len(outliers) > 0:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        col1.write(f"**{col}** ({len(outliers)} outliers)")
                        method = col2.selectbox("Método", ['Não tratar', 'Winsorização', 'Log Transform'], key=f"outlier_{col}")
                        if col3.checkbox("Aplicar", key=f"apply_outlier_{col}") and method != 'Não tratar':
                            if method == 'Winsorização':
                                outlier_treatment[col] = 'winsorize'
                            elif method == 'Log Transform':
                                outlier_treatment[col] = 'log_transform'
                
                if outlier_treatment:
                    corrections_config['outlier_treatment'] = outlier_treatment
                
                # Criação de variáveis derivadas
                if report['sugestoes_fusao']:
                    st.write("**🔗 Criação de Variáveis Derivadas:**")
                    create_derived = []
                    for suggestion in report['sugestoes_fusao']:
                        if suggestion['tipo'] == 'variavel_derivada':
                            with st.expander(f"Sugestão: {suggestion['sugestao']}"):
                                st.write(f"**Motivo:** {suggestion['motivo']}")
                                if st.checkbox(f"Criar variável derivada", key=f"derived_{suggestion['variaveis'][0]}"):
                                    if 'idade' in suggestion['variaveis']:
                                        create_derived.append({'type': 'age_groups', 'column': 'idade'})
                                    elif 'renda' in suggestion['variaveis']:
                                        create_derived.append({'type': 'income_log', 'column': 'renda'})
                    
                    if create_derived:
                        corrections_config['create_derived'] = create_derived
                
                # Botão para aplicar todas as correções
                if st.button("🚀 Aplicar Todas as Correções Selecionadas"):
                    if corrections_config:
                        with st.spinner("Aplicando correções..."):
                            df_corrected = ai.apply_data_corrections(df, corrections_config)
                            st.session_state.df_corrected = df_corrected
                            st.session_state.corrections_applied = corrections_config
                            st.success("✅ Correções aplicadas com sucesso!")
                            
                            # Mostrar resumo das mudanças
                            st.subheader("📋 Resumo das Correções Aplicadas:")
                            for key, value in corrections_config.items():
                                if key == 'rename_columns':
                                    st.write(f"• **Colunas renomeadas:** {len(value)} colunas")
                                elif key == 'missing_treatment':
                                    st.write(f"• **Dados faltantes tratados:** {len(value)} colunas")
                                elif key == 'standardize_values':
                                    st.write(f"• **Valores padronizados:** {len(value)} colunas")
                                elif key == 'outlier_treatment':
                                    st.write(f"• **Outliers tratados:** {len(value)} colunas")
                                elif key == 'create_derived':
                                    st.write(f"• **Variáveis derivadas criadas:** {len(value)} variáveis")
                    else:
                        st.warning("Nenhuma correção foi selecionada!")
                
                # Mostrar comparação antes/depois
                if 'df_corrected' in st.session_state:
                    st.subheader("📊 Comparação: Antes vs. Depois")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset Original:**")
                        st.write(f"• Linhas: {df.shape[0]}")
                        st.write(f"• Colunas: {df.shape[1]}")
                        st.write(f"• Missing total: {df.isnull().sum().sum()}")
                        st.write(f"• Tipos de dados: {df.dtypes.value_counts().to_dict()}")
                    
                    with col2:
                        st.write("**Dataset Corrigido:**")
                        df_corr = st.session_state.df_corrected
                        st.write(f"• Linhas: {df_corr.shape[0]}")
                        st.write(f"• Colunas: {df_corr.shape[1]}")
                        st.write(f"• Missing total: {df_corr.isnull().sum().sum()}")
                        st.write(f"• Tipos de dados: {df_corr.dtypes.value_counts().to_dict()}")
                    
                    # Opção de usar dataset corrigido nas próximas análises
                    if st.checkbox("🔄 Usar dataset corrigido nas próximas análises"):
                        st.session_state.df_to_use = st.session_state.df_corrected
                        st.info("✅ Dataset corrigido será usado nas próximas análises!")
                    else:
                        st.session_state.df_to_use = df
            
            else:
                st.info("👆 Clique em 'Analisar Qualidade dos Dados' para começar")
        
        # Usar dataset corrigido se disponível
        df_analysis = st.session_state.get('df_to_use', df)
        
        with tab1:
            st.header("📊 Análise Univariada com IA")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_var = st.selectbox("Selecione uma variável:", df_analysis.columns)
                
                if st.button("🔍 Analisar com IA"):
                    analysis = ai.analyze_univariate(df_analysis, selected_var)
                    
                    st.subheader(f"Análise: {selected_var}")
                    st.write(f"**Tipo:** {analysis['tipo']}")
                    
                    # Estatísticas
                    stats_df = pd.DataFrame(analysis['estatisticas'], index=[0]).T
                    st.write("**Estatísticas:**")
                    st.dataframe(stats_df)
                    
                    # Sugestão da IA
                    st.write("**💡 Sugestão da IA:**")
                    st.info(analysis['sugestao_ia'])
            
            with col2:
                if 'selected_var' in locals():
                    # Gráfico
                    if df_analysis[selected_var].dtype in ['int64', 'float64']:
                        fig = px.histogram(df_analysis, x=selected_var, title=f"Distribuição de {selected_var}")
                    else:
                        fig = px.bar(df_analysis[selected_var].value_counts().reset_index(), 
                                   x='index', y=selected_var, title=f"Frequência de {selected_var}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("🔗 Análise Bivariada com IA")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                var1 = st.selectbox("Primeira variável:", df_analysis.columns)
                var2 = st.selectbox("Segunda variável:", df_analysis.columns)
                
                if st.button("🔍 Análise Bivariada"):
                    analysis = ai.analyze_bivariate(df_analysis, var1, var2, target_col)
                    
                    st.subheader(f"Análise: {var1} vs {var2}")
                    
                    if 'correlacao' in analysis:
                        st.metric("Correlação", f"{analysis['correlacao']:.3f}")
                    
                    if 'correlacao_target' in analysis:
                        st.metric("Correlação com Target", f"{analysis['correlacao_target']:.3f}")
                    
                    st.write("**💡 Insights da IA:**")
                    st.info(analysis['sugestao_ia'])
            
            with col2:
                if 'var1' in locals() and 'var2' in locals():
                    if df_analysis[var1].dtype in ['int64', 'float64'] and df_analysis[var2].dtype in ['int64', 'float64']:
                        fig = px.scatter(df_analysis, x=var1, y=var2, color=target_col if target_col else None)
                    else:
                        # Crosstab para categóricas
                        crosstab = pd.crosstab(df_analysis[var1], df_analysis[var2])
                        fig = px.imshow(crosstab, text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)= st.selectbox("Primeira variável:", df.columns)
                var2 = st.selectbox("Segunda variável:", df.columns)
                
                if st.button("🔍 Análise Bivariada"):
                    analysis = ai.analyze_bivariate(df, var1, var2, target_col)
                    
                    st.subheader(f"Análise: {var1} vs {var2}")
                    
                    if 'correlacao' in analysis:
                        st.metric("Correlação", f"{analysis['correlacao']:.3f}")
                    
                    if 'correlacao_target' in analysis:
                        st.metric("Correlação com Target", f"{analysis['correlacao_target']:.3f}")
                    
                    st.write("**💡 Insights da IA:**")
                    st.info(analysis['sugestao_ia'])
            
            with col2:
                if 'var1' in locals() and 'var2' in locals():
                    if df[var1].dtype in ['int64', 'float64'] and df[var2].dtype in ['int64', 'float64']:
                        fig = px.scatter(df, x=var1, y=var2, color=target_col if target_col else None)
                    else:
                        # Crosstab para categóricas
                        crosstab = pd.crosstab(df[var1], df[var2])
                        fig = px.imshow(crosstab, text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("🤖 Estratégia de Modelagem")
            
            # Sugestões estratégicas
            suggestions = ai.suggest_model_strategy(df, target_col)
            
            st.subheader("💡 Sugestões da IA para sua modelagem:")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
            
            # Seleção de modelo
            st.subheader("🎯 Recomendações de Modelos:")
            for model, description in ai.model_suggestions.items():
                st.write(f"**{model.replace('_', ' ').title()}:** {description}")
            
            # Botão para executar modelagem simples
            if st.button("🚀 Executar Modelagem Demonstrativa"):
                with st.spinner("Treinando modelo..."):
                    # Preparação simplificada dos dados
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if target_col in numeric_cols:
                        numeric_cols.remove(target_col)
                    
                    X = df[numeric_cols].fillna(df[numeric_cols].median())
                    y = df[target_col]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    # Modelo simples
                    model = LogisticRegression(random_state=42)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Salvar resultados
                    st.session_state.model_results = {
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'model': model,
                        'feature_names': numeric_cols
                    }
                    
                    st.success("✅ Modelo treinado com sucesso!")
        
        with tab4:
            st.header("📈 Análise de Performance")
            
            if 'model_results' in st.session_state:
                results = st.session_state.model_results
                
                # Insights da IA
                insights = ai.generate_performance_insights(
                    results['y_test'], 
                    results['y_pred'], 
                    results['y_proba']
                )
                
                st.subheader("💡 Insights da IA sobre Performance:")
                for insight in insights:
                    st.write(f"• {insight}")
                
                # Métricas visuais
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROC Curve
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(results['y_test'], results['y_proba'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
                    fig.update_layout(title='Curva ROC', xaxis_title='Taxa de Falsos Positivos', yaxis_title='Taxa de Verdadeiros Positivos')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Matriz de Confusão
                    cm = confusion_matrix(results['y_test'], results['y_pred'])
                    fig = px.imshow(cm, text_auto=True, title='Matriz de Confusão')
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Execute a modelagem primeiro na aba 'Modelagem'")
        
        with tab5:
            st.header("📋 Políticas de Crédito")
            
            if 'model_results' in st.session_state:
                # Sugestões de políticas
                policies = ai.suggest_credit_policies(
                    st.session_state.model_results, 
                    business_context
                )
                
                st.subheader("🤖 Recomendações da IA:")
                for policy in policies:
                    if policy.startswith('🎯') or policy.startswith('📋'):
                        st.subheader(policy)
                    else:
                        st.write(policy)
                
                # Simulador de políticas
                st.subheader("🎮 Simulador de Políticas")
                
                threshold = st.slider("Threshold de aprovação:", 0.0, 1.0, 0.5, 0.01)
                
                if st.button("🔍 Simular Política"):
                    y_pred_policy = (st.session_state.model_results['y_proba'] >= threshold).astype(int)
                    
                    # Métricas da política
                    from sklearn.metrics import precision_score, recall_score
                    precision = precision_score(st.session_state.model_results['y_test'], y_pred_policy)
                    recall = recall_score(st.session_state.model_results['y_test'], y_pred_policy)
                    approval_rate = y_pred_policy.mean()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Taxa de Aprovação", f"{approval_rate:.1%}")
                    col2.metric("Precisão", f"{precision:.3f}")
                    col3.metric("Recall", f"{recall:.3f}")
                    
                    st.info(f"💡 Com threshold {threshold:.2f}, você aprovaria {approval_rate:.1%} dos clientes com precisão de {precision:.3f}")
            
            else:
                st.info("Execute a modelagem primeiro para gerar recomendações de políticas")
    
    else:
        st.info("👆 Faça upload de um dataset CSV para começar a análise")
        
        # Exemplo de dataset
        st.subheader("📝 Formato esperado do dataset:")
        example_data = {
            'idade': [25, 35, 45, 30],
            'renda': [3000, 5000, 8000, 4000],
            'score_bureau': [650, 720, 800, 680],
            'tempo_emprego': [12, 36, 60, 24],
            'inadimplente': [0, 0, 0, 1]
        }
        st.dataframe(pd.DataFrame(example_data))
        st.caption("* A coluna 'inadimplente' deve ser binária (0=bom pagador, 1=mau pagador)")

if __name__ == "__main__":
    main()
