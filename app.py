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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Análise Univariada", "🔗 Análise Bivariada", "🤖 Modelagem", "📈 Performance", "📋 Políticas"])
        
        with tab1:
            st.header("📊 Análise Univariada com IA")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_var = st.selectbox("Selecione uma variável:", df.columns)
                
                if st.button("🔍 Analisar com IA"):
                    analysis = ai.analyze_univariate(df, selected_var)
                    
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
                    if df[selected_var].dtype in ['int64', 'float64']:
                        fig = px.histogram(df, x=selected_var, title=f"Distribuição de {selected_var}")
                    else:
                        fig = px.bar(df[selected_var].value_counts().reset_index(), 
                                   x='index', y=selected_var, title=f"Frequência de {selected_var}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("🔗 Análise Bivariada com IA")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                var1 = st.selectbox("Primeira variável:", df.columns)
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
