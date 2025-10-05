#!/usr/bin/env python3
"""
Generate comprehensive PDF reports for hedge fund due diligence.
Based on case study requirements for BDPRF hedge fund analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Add src to path for performance_metrics
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from performance_metrics import summary_table, cumulative_returns, drawdown_series

class FundAnalysisPDF:
    def __init__(self, fund_name, fund_data, benchmark_data, summary_stats):
        self.fund_name = fund_name
        self.fund_data = fund_data
        self.benchmark_data = benchmark_data
        self.summary_stats = summary_stats
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkred
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=8,
            textColor=colors.darkgreen
        ))
        
        # Add Python code style with background color
        self.styles.add(ParagraphStyle(
            name='PythonCode',
            parent=self.styles['Code'],
            fontSize=9,
            fontName='Courier',
            backColor=colors.lightgrey,
            borderColor=colors.grey,
            borderWidth=1,
            borderPadding=6,
            leftIndent=12,
            rightIndent=12,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        # Add mathematical formula style
        self.styles.add(ParagraphStyle(
            name='MathFormula',
            parent=self.styles['Normal'],
            fontSize=12,
            fontName='Times-Roman',
            alignment=TA_CENTER,
            spaceBefore=8,
            spaceAfter=16,
            backColor=colors.white,
            borderColor=colors.black,
            borderWidth=0.5,
            borderPadding=8
        ))

    def create_title_page(self, story):
        """Create title page."""
        story.append(Spacer(1, 2*inch))
        
        title = Paragraph(f"Hedge Fund Due Diligence Report", self.styles['CustomTitle'])
        story.append(title)
        
        story.append(Spacer(1, 0.5*inch))
        
        subtitle = Paragraph(f"{self.fund_name}", self.styles['CustomTitle'])
        story.append(subtitle)
        
        story.append(Spacer(1, 0.3*inch))
        
        date_text = Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", 
                             self.styles['Normal'])
        story.append(date_text)
        
        story.append(Spacer(1, 0.5*inch))
        
        # Add group members
        members_text = """
        <b>Group Members:</b><br/>
        Paul Garofalo, Sebastian Lopez-Irizarry, Tao Xie, and Dominik Dimitrov
        """
        story.append(Paragraph(members_text, self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        purpose = Paragraph(
            "Prepared for: The New York Bus Driver Pension and Relief Fund (BDPRF)<br/>"
            "Purpose: Due diligence analysis for potential hedge fund allocation",
            self.styles['Normal']
        )
        story.append(purpose)
        
        story.append(PageBreak())

    def create_executive_summary(self, story):
        """Create executive summary section."""
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        # Key metrics summary
        cagr = self.summary_stats.get('CAGR', 0)
        sharpe = self.summary_stats.get('Sharpe', 0)
        max_dd = self.summary_stats.get('MaxDrawdown', 0)
        calmar = self.summary_stats.get('Calmar', 0)
        alpha = self.summary_stats.get('Alpha_ann', 0)
        beta = self.summary_stats.get('Beta', 0)
        
        summary_text = f"""
        <b>Key Performance Metrics:</b><br/>
        • Annualized Return (CAGR): {cagr:.1%}<br/>
        • Sharpe Ratio: {sharpe:.2f}<br/>
        • Maximum Drawdown: {max_dd:.1%}<br/>
        • Calmar Ratio: {calmar:.2f}<br/>
        • Alpha (vs S&P 500): {alpha:.1%}<br/>
        • Beta (vs S&P 500): {beta:.2f}<br/><br/>
        
        <b>Investment Recommendation:</b><br/>
        Based on quantitative analysis, {self.fund_name} demonstrates {'strong' if sharpe > 1.0 else 'moderate'} 
        risk-adjusted returns with a Sharpe ratio of {sharpe:.2f}. The fund has generated 
        {'positive' if alpha > 0 else 'negative'} alpha of {alpha:.1%} relative to the S&P 500 benchmark.
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

    def _get_metric_notation(self, metric_name):
        """Get mathematical notation and code reference for a metric."""
        notations = {
            'CAGR': {
                'formula': r'$CAGR = \left(\prod_{i=1}^{T}(1 + r_i)\right)^{1/T} - 1$',
                'code': 'cagr(r, ppy) in performance_metrics.py',
                'description': 'Compound Annual Growth Rate'
            },
            'AnnVol': {
                'formula': r'$\sigma_{ann} = \sigma_{period} \times \sqrt{periods\_per\_year}$',
                'code': 'annualized_vol(r, ppy) in performance_metrics.py',
                'description': 'Annualized Volatility'
            },
            'Sharpe': {
                'formula': r'$Sharpe = \frac{\mu - r_f}{\sigma}$',
                'code': 'sharpe(r, rf, ppy) in performance_metrics.py',
                'description': 'Sharpe Ratio'
            },
            'Sortino': {
                'formula': r'$Sortino = \frac{\mu - r_f}{\sigma_{downside}}$',
                'code': 'sortino(r, rf, ppy, mar) in performance_metrics.py',
                'description': 'Sortino Ratio'
            },
            'MaxDrawdown': {
                'formula': r'$MDD = \min\left(\frac{W_t}{W_{peak}} - 1\right)$',
                'code': 'max_drawdown(r) in performance_metrics.py',
                'description': 'Maximum Drawdown'
            },
            'Calmar': {
                'formula': r'$Calmar = \frac{CAGR}{|MDD|}$',
                'code': 'calmar(r, ppy) in performance_metrics.py',
                'description': 'Calmar Ratio'
            },
            'HitRatio': {
                'formula': r'$Hit\ Ratio = \frac{positive\ periods}{total\ periods}$',
                'code': 'hit_ratio(r) in performance_metrics.py',
                'description': 'Hit Ratio'
            },
            'Alpha_ann': {
                'formula': r'$\alpha_{ann} = (1 + \alpha_{period})^{periods\_per\_year} - 1$',
                'code': 'alpha_beta(r, bench, ppy) in performance_metrics.py',
                'description': 'Annualized Alpha'
            },
            'Beta': {
                'formula': r'$\beta = \frac{Cov(r, r_{bench})}{Var(r_{bench})}$',
                'code': 'alpha_beta(r, bench, ppy) in performance_metrics.py',
                'description': 'Beta'
            },
            'R2': {
                'formula': r'$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$',
                'code': 'alpha_beta(r, bench, ppy) in performance_metrics.py',
                'description': 'R-squared'
            },
            'InformationRatio': {
                'formula': r'$IR = \frac{\mu - \mu_{bench}}{\sigma(\mu - \mu_{bench})}$',
                'code': 'information_ratio(r, bench, ppy) in performance_metrics.py',
                'description': 'Information Ratio'
            },
            'VaR_95': {
                'formula': r'$VaR_{95\%} = 5^{th}\ percentile\ of\ returns$',
                'code': 'var_historical(r, 0.95) in performance_metrics.py',
                'description': 'Value at Risk (95%)'
            },
            'CVaR_95': {
                'formula': r'$CVaR_{95\%} = E[r | r \leq VaR_{95\%}]$',
                'code': 'cvar_historical(r, 0.95) in performance_metrics.py',
                'description': 'Conditional Value at Risk (95%)'
            }
        }
        return notations.get(metric_name, {'formula': 'N/A', 'code': 'N/A', 'description': 'N/A'})

    def create_quantitative_analysis(self, story):
        """Create quantitative analysis section."""
        story.append(Paragraph("QUANTITATIVE ANALYSIS", self.styles['SectionHeader']))
        
        # Individual metric explanations
        self._add_metric_explanations(story)
        
        # Analysis text
        analysis_text = f"""
        <b>Risk-Reward Analysis:</b><br/>
        The fund demonstrates {'strong' if self.summary_stats.get('Sharpe', 0) > 1.0 else 'moderate'} 
        risk-adjusted performance with a Sharpe ratio of {self.summary_stats.get('Sharpe', 0):.2f}. 
        {'Positive' if self.summary_stats.get('Alpha_ann', 0) > 0 else 'Negative'} alpha generation 
        of {self.summary_stats.get('Alpha_ann', 0):.1%} indicates {'superior' if self.summary_stats.get('Alpha_ann', 0) > 0 else 'inferior'} 
        risk-adjusted returns compared to the S&P 500 benchmark.<br/><br/>
        
        <b>Risk Characteristics:</b><br/>
        • Maximum drawdown of {self.summary_stats.get('MaxDrawdown', 0):.1%} {'exceeds' if abs(self.summary_stats.get('MaxDrawdown', 0)) > 0.15 else 'is within acceptable limits for'} 
        typical hedge fund standards<br/>
        • Beta of {self.summary_stats.get('Beta', 0):.2f} indicates {'high' if self.summary_stats.get('Beta', 0) > 0.8 else 'moderate' if self.summary_stats.get('Beta', 0) > 0.5 else 'low'} 
        correlation with market movements<br/>
        • R-squared of {self.summary_stats.get('R2', 0):.1%} suggests {'strong' if self.summary_stats.get('R2', 0) > 0.5 else 'weak'} 
        explanatory power of market factors
        """
        
        story.append(Paragraph(analysis_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))


    def _add_metric_explanations(self, story):
        """Add individual metric explanations with LaTeX notation and code."""
        story.append(Paragraph("<b>Performance Metrics Analysis</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # CAGR
        cagr_val = self.summary_stats.get('CAGR', 0)
        cagr_text = f"""
        <b>1. Compound Annual Growth Rate (CAGR): {cagr_val:.1%}</b><br/>
        <b>Mathematical Formula:</b><br/>
        """
        story.append(Paragraph(cagr_text, self.styles['Normal']))
        
        # Add mathematical formula with proper styling
        formula_text = "CAGR = (∏(1 + r<sub>i</sub>))<sup>1/T</sup> - 1"
        story.append(Paragraph(formula_text, self.styles['MathFormula']))
        story.append(Spacer(1, 0.1*inch))  # Add extra space after formula
        
        # Add Python implementation with background
        python_text = f"""
        <b>Python Implementation:</b><br/>
        def cagr(r, ppy=None):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;r = pd.Series(r).dropna()<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;if r.empty: return np.nan<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;ppy = ppy or infer_ppy(r.index)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;tot = (1+r).prod()<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;yrs = len(r)/ppy<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;return tot**(1/yrs)-1
        """
        story.append(Paragraph(python_text, self.styles['PythonCode']))
        
        # Add interpretation
        interpretation_text = f"<b>Interpretation:</b> {self.fund_name} has generated an annualized return of {cagr_val:.1%} over the analysis period. This {'exceeds' if cagr_val > 0.10 else 'meets' if cagr_val > 0.05 else 'falls below'} typical hedge fund return expectations, indicating {'strong' if cagr_val > 0.10 else 'moderate' if cagr_val > 0.05 else 'weak'} long-term performance."
        story.append(Paragraph(interpretation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Sharpe Ratio
        sharpe_val = self.summary_stats.get('Sharpe', 0)
        sharpe_text = f"""
        <b>2. Sharpe Ratio: {sharpe_val:.2f}</b><br/>
        <b>Mathematical Formula:</b><br/>
        """
        story.append(Paragraph(sharpe_text, self.styles['Normal']))
        
        # Add mathematical formula with proper styling
        formula_text = """
        Sharpe = <table border="0" cellpadding="0" cellspacing="0" style="display:inline-table; vertical-align:middle;">
        <tr><td align="center" style="border-bottom:1px solid black; padding:0 2px;">μ - r<sub>f</sub></td></tr>
        <tr><td align="center" style="padding:2px 2px 0 2px;">σ</td></tr>
        </table>
        """
        story.append(Paragraph(formula_text, self.styles['MathFormula']))
        story.append(Spacer(1, 0.1*inch))  # Add extra space after formula
        
        # Add Python implementation with background
        python_text = f"""
        <b>Python Implementation:</b><br/>
        def sharpe(r, rf=0.02, ppy=None):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;r = pd.Series(r).dropna()<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;ppy = ppy or infer_ppy(r.index)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;rf_per = (1+rf)**(1/ppy)-1<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;ex = r - rf_per<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;ret = cagr(ex, ppy)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;vol = ann_vol(ex, ppy)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;return ret/vol
        """
        story.append(Paragraph(python_text, self.styles['PythonCode']))
        
        # Add interpretation
        interpretation_text = f"<b>Interpretation:</b> {self.fund_name} demonstrates {'excellent' if sharpe_val > 1.5 else 'good' if sharpe_val > 1.0 else 'fair' if sharpe_val > 0.5 else 'poor'} risk-adjusted performance with a Sharpe ratio of {sharpe_val:.2f}. This indicates the fund {'effectively' if sharpe_val > 1.0 else 'moderately' if sharpe_val > 0.5 else 'ineffectively'} generates returns relative to the risk taken, suggesting {'strong' if sharpe_val > 1.0 else 'moderate' if sharpe_val > 0.5 else 'weak'} risk management capabilities."
        story.append(Paragraph(interpretation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Maximum Drawdown
        mdd_val = self.summary_stats.get('MaxDrawdown', 0)
        mdd_text = f"""
        <b>3. Maximum Drawdown: {mdd_val:.1%}</b><br/>
        <b>Mathematical Formula:</b><br/>
        """
        story.append(Paragraph(mdd_text, self.styles['Normal']))
        
        # Add mathematical formula with proper styling
        formula_text = """
        MDD = min(<table border="0" cellpadding="0" cellspacing="0" style="display:inline-table; vertical-align:middle;">
        <tr><td align="center" style="border-bottom:1px solid black; padding:0 2px;">W<sub>t</sub></td></tr>
        <tr><td align="center" style="padding:2px 2px 0 2px;">W<sub>peak</sub></td></tr>
        </table> - 1)
        """
        story.append(Paragraph(formula_text, self.styles['MathFormula']))
        story.append(Spacer(1, 0.1*inch))  # Add extra space after formula
        
        # Add Python implementation with background
        python_text = f"""
        <b>Python Implementation:</b><br/>
        def drawdown_series(r):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;wealth = (1+r).cumprod()<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;peak = wealth.cummax()<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;return wealth/peak - 1.0<br/><br/>
        def max_drawdown(r):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;dd = drawdown_series(r)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;return dd.min()
        """
        story.append(Paragraph(python_text, self.styles['PythonCode']))
        
        # Add interpretation
        interpretation_text = f"<b>Interpretation:</b> {self.fund_name} experienced a maximum drawdown of {mdd_val:.1%}, which represents {'significant' if abs(mdd_val) > 0.20 else 'moderate' if abs(mdd_val) > 0.10 else 'limited'} downside risk. This {'exceeds' if abs(mdd_val) > 0.15 else 'is within' if abs(mdd_val) > 0.10 else 'is below'} typical hedge fund drawdown thresholds, indicating {'concerning' if abs(mdd_val) > 0.15 else 'acceptable' if abs(mdd_val) > 0.10 else 'excellent'} risk control by the fund's management team."
        story.append(Paragraph(interpretation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Calmar Ratio
        calmar_val = self.summary_stats.get('Calmar', 0)
        calmar_text = f"""
        <b>4. Calmar Ratio: {calmar_val:.2f}</b><br/>
        <b>Mathematical Formula:</b><br/>
        """
        story.append(Paragraph(calmar_text, self.styles['Normal']))
        
        # Add mathematical formula with proper styling
        formula_text = """
        Calmar = <table border="0" cellpadding="0" cellspacing="0" style="display:inline-table; vertical-align:middle;">
        <tr><td align="center" style="border-bottom:1px solid black; padding:0 2px;">CAGR</td></tr>
        <tr><td align="center" style="padding:2px 2px 0 2px;">|MDD|</td></tr>
        </table>
        """
        story.append(Paragraph(formula_text, self.styles['MathFormula']))
        story.append(Spacer(1, 0.1*inch))  # Add extra space after formula
        
        # Add Python implementation with background
        python_text = f"""
        <b>Python Implementation:</b><br/>
        def calmar(r, ppy=None):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;mdd = max_drawdown(r)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;if mdd == 0 or np.isnan(mdd):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return np.nan<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;return cagr(r, ppy) / abs(mdd)
        """
        story.append(Paragraph(python_text, self.styles['PythonCode']))
        
        # Add interpretation
        interpretation_text = f"<b>Interpretation:</b> {self.fund_name} achieves a Calmar ratio of {calmar_val:.2f}, demonstrating {'exceptional' if calmar_val > 2.0 else 'strong' if calmar_val > 1.0 else 'moderate' if calmar_val > 0.5 else 'weak'} risk-adjusted performance relative to maximum drawdown. This indicates the fund {'effectively' if calmar_val > 1.0 else 'moderately' if calmar_val > 0.5 else 'ineffectively'} balances return generation with downside risk control, suggesting {'superior' if calmar_val > 1.0 else 'adequate' if calmar_val > 0.5 else 'inferior'} risk management and capital preservation capabilities."
        story.append(Paragraph(interpretation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Alpha
        alpha_val = self.summary_stats.get('Alpha_ann', 0)
        alpha_text = f"""
        <b>5. Alpha (Annualized): {alpha_val:.1%}</b><br/>
        <b>Mathematical Formula:</b><br/>
        """
        story.append(Paragraph(alpha_text, self.styles['Normal']))
        
        # Add mathematical formula with proper styling
        formula_text = "α = (1 + α<sub>period</sub>)<sup>periods_per_year</sup> - 1"
        story.append(Paragraph(formula_text, self.styles['MathFormula']))
        story.append(Spacer(1, 0.1*inch))  # Add extra space after formula
        
        # Add Python implementation with background
        python_text = f"""
        <b>Python Implementation:</b><br/>
        def alpha_beta(r, bench, ppy=None):<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;r = _clean_returns(r)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;b = _clean_returns(bench)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;df = pd.concat([r, b], axis=1).dropna()<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;y = df.iloc[:, 0].values<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;x = df.iloc[:, 1].values<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;x1 = np.column_stack((np.ones(len(x)), x))<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;beta_hat = np.linalg.lstsq(x1, y, rcond=None)[0]<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;a, b_ = beta_hat[0], beta_hat[1]<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;alpha_ann = (1 + a) ** ppy - 1
        """
        story.append(Paragraph(python_text, self.styles['PythonCode']))
        
        # Add interpretation
        interpretation_text = f"<b>Interpretation:</b> {self.fund_name} generates {'positive' if alpha_val > 0 else 'negative'} alpha of {alpha_val:.1%} annually, indicating {'superior' if alpha_val > 0.05 else 'moderate' if alpha_val > 0.02 else 'weak' if alpha_val > 0 else 'negative'} investment skill relative to the S&P 500 benchmark. This {'demonstrates' if alpha_val > 0 else 'lacks'} the fund's ability to generate excess returns through {'effective' if alpha_val > 0.02 else 'limited' if alpha_val > 0 else 'ineffective'} stock selection and market timing, suggesting {'strong' if alpha_val > 0.05 else 'moderate' if alpha_val > 0 else 'weak'} active management capabilities."
        story.append(Paragraph(interpretation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Beta
        beta_val = self.summary_stats.get('Beta', 0)
        beta_text = f"""
        <b>6. Beta: {beta_val:.2f}</b><br/>
        <b>Mathematical Formula:</b><br/>
        """
        story.append(Paragraph(beta_text, self.styles['Normal']))
        
        # Add mathematical formula with proper styling
        formula_text = """
        β = <table border="0" cellpadding="0" cellspacing="0" style="display:inline-table; vertical-align:middle;">
        <tr><td align="center" style="border-bottom:1px solid black; padding:0 2px;">Cov(r, r<sub>bench</sub>)</td></tr>
        <tr><td align="center" style="padding:2px 2px 0 2px;">Var(r<sub>bench</sub>)</td></tr>
        </table>
        """
        story.append(Paragraph(formula_text, self.styles['MathFormula']))
        story.append(Spacer(1, 0.1*inch))  # Add extra space after formula
        
        # Add Python implementation with background
        python_text = f"""
        <b>Python Implementation:</b><br/>
        # Calculated in alpha_beta function above<br/>
        beta = beta_hat[1]  # Second coefficient from regression
        """
        story.append(Paragraph(python_text, self.styles['PythonCode']))
        
        # Add interpretation
        interpretation_text = f"<b>Interpretation:</b> {self.fund_name} exhibits a beta of {beta_val:.2f}, indicating {'high' if beta_val > 0.8 else 'moderate' if beta_val > 0.5 else 'low'} sensitivity to market movements. This suggests the fund {'closely' if beta_val > 0.8 else 'moderately' if beta_val > 0.5 else 'minimally'} follows market trends, demonstrating {'high' if beta_val > 0.8 else 'moderate' if beta_val > 0.5 else 'low'} systematic risk exposure. The fund's {'aggressive' if beta_val > 1.0 else 'defensive' if beta_val < 0.5 else 'balanced'} positioning relative to the market indicates {'strong' if beta_val > 0.8 else 'moderate' if beta_val > 0.5 else 'weak'} correlation with broader market movements."
        story.append(Paragraph(interpretation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

    def create_investment_due_diligence(self, story):
        """Create investment due diligence section."""
        story.append(Paragraph("INVESTMENT DUE DILIGENCE", self.styles['SectionHeader']))
        
        # Fund-specific analysis based on case study data
        if "Fund 1" in self.fund_name:
            self._add_fund1_investment_analysis(story)
        elif "Fund 2" in self.fund_name:
            self._add_fund2_investment_analysis(story)
        elif "Fund 3" in self.fund_name:
            self._add_fund3_investment_analysis(story)
        
        story.append(Spacer(1, 0.2*inch))

    def _add_fund1_investment_analysis(self, story):
        """Add Fund 1 specific investment analysis."""
        analysis_text = """
        <b>Investment Strategy Assessment:</b><br/>
        • <b>Strengths:</b> Long-term fundamental value approach, experienced PM with 20+ years experience, 
        low turnover strategy with 24-36 month holding periods, focus on small to mid-cap opportunities<br/>
        • <b>Weaknesses:</b> Single PM dependency, limited diversification (8-10 positions), 
        high concentration risk, no systematic risk management<br/><br/>
        
        <b>Performance Attribution:</b><br/>
        The fund's performance is driven by stock selection rather than market timing. 
        The fundamental research approach has generated consistent alpha through deep 
        company analysis and long-term value creation.<br/><br/>
        
        <b>Risk Management Concerns:</b><br/>
        • No dedicated risk manager - PM assumes dual role<br/>
        • Soft stop-loss at -50% may be too lenient<br/>
        • High position concentration (10% per position)<br/>
        • Limited hedging capabilities (ETFs/indices only)
        """
        story.append(Paragraph(analysis_text, self.styles['Normal']))

    def _add_fund2_investment_analysis(self, story):
        """Add Fund 2 specific investment analysis."""
        analysis_text = """
        <b>Investment Strategy Assessment:</b><br/>
        • <b>Strengths:</b> Global equity focus with developed/EM coverage, academic research environment, 
        experienced PM from top-tier fund, diversified team structure<br/>
        • <b>Weaknesses:</b> Limited track record (launched 2012), high fees (3.5% management, 35% performance), 
        quarterly redemptions with 45-day notice, 12-month lockup<br/><br/>
        
        <b>Performance Attribution:</b><br/>
        The fund benefits from global diversification and the PM's experience from a 
        large institutional fund. However, the track record is relatively short and 
        includes a hypothetical period.<br/><br/>
        
        <b>Risk Management Assessment:</b><br/>
        • Discretionary risk management by PM only<br/>
        • No systematic hedging approach<br/>
        • High gross exposure limits (175% max)<br/>
        • Liquidity requirements may limit investment flexibility
        """
        story.append(Paragraph(analysis_text, self.styles['Normal']))

    def _add_fund3_investment_analysis(self, story):
        """Add Fund 3 specific investment analysis."""
        analysis_text = """
        <b>Investment Strategy Assessment:</b><br/>
        • <b>Strengths:</b> Multi-PM structure with sector specialists, institutional-grade infrastructure, 
        dedicated risk management team, proven track record since 2005<br/>
        • <b>Weaknesses:</b> High employee turnover, complex organizational structure, 
        potential for style drift across PMs<br/><br/>
        
        <b>Performance Attribution:</b><br/>
        The fund's performance benefits from sector specialization and experienced PMs. 
        The multi-PM structure provides diversification but may lead to inconsistent 
        performance across sectors.<br/><br/>
        
        <b>Risk Management Strengths:</b><br/>
        • Dedicated risk management team with real-time monitoring<br/>
        • Systematic risk limits and controls<br/>
        • Comprehensive risk reporting and attribution analysis<br/>
        • Professional risk infrastructure and systems
        """
        story.append(Paragraph(analysis_text, self.styles['Normal']))

    def create_operational_due_diligence(self, story):
        """Create operational due diligence section."""
        story.append(Paragraph("OPERATIONAL DUE DILIGENCE", self.styles['SectionHeader']))
        
        if "Fund 1" in self.fund_name:
            self._add_fund1_operational_analysis(story)
        elif "Fund 2" in self.fund_name:
            self._add_fund2_operational_analysis(story)
        elif "Fund 3" in self.fund_name:
            self._add_fund3_operational_analysis(story)
        
        story.append(Spacer(1, 0.2*inch))
        
        # Add performance charts section
        self._add_performance_charts_section(story)

    def _add_fund1_operational_analysis(self, story):
        """Add Fund 1 operational analysis."""
        analysis_text = """
        <b>Operational Strengths:</b><br/>
        • Simple organizational structure with clear segregation of duties<br/>
        • Reputable service providers (prime broker, administrator, legal)<br/>
        • Experienced team with long tenure<br/>
        • No regulatory issues or compliance concerns<br/><br/>
        
        <b>Operational Concerns:</b><br/>
        • <b>Key Person Risk:</b> Heavy reliance on single PM for all investment decisions<br/>
        • <b>Succession Planning:</b> No clear succession plan for PM role<br/>
        • <b>Infrastructure:</b> Outsourced IT and legal functions may create dependencies<br/>
        • <b>Scalability:</b> Limited team size may constrain growth<br/><br/>
        
        <b>Client Base Analysis:</b><br/>
        • High concentration in fund-of-funds (50% of AUM)<br/>
        • Top client represents 38.89% of fund assets<br/>
        • Limited institutional diversification
        """
        story.append(Paragraph(analysis_text, self.styles['Normal']))

    def _add_fund2_operational_analysis(self, story):
        """Add Fund 2 operational analysis."""
        analysis_text = """
        <b>Operational Strengths:</b><br/>
        • SEC registered with proper regulatory compliance<br/>
        • Reputable service providers across all functions<br/>
        • Strong team with relevant experience<br/>
        • Academic research environment promotes collaboration<br/><br/>
        
        <b>Operational Concerns:</b><br/>
        • <b>Track Record:</b> Limited actual performance history (launched 2012)<br/>
        • <b>Transparency:</b> Limited disclosure on client base and largest clients<br/>
        • <b>Fees:</b> High fee structure (3.5% management, 35% performance)<br/>
        • <b>Liquidity:</b> Quarterly redemptions with 45-day notice and 12-month lockup<br/><br/>
        
        <b>Risk Factors:</b><br/>
        • PM's previous fund failed during 2008 crisis<br/>
        • Hypothetical track record period raises questions<br/>
        • High concentration risk in early-stage fund
        """
        story.append(Paragraph(analysis_text, self.styles['Normal']))

    def _add_fund3_operational_analysis(self, story):
        """Add Fund 3 operational analysis."""
        analysis_text = """
        <b>Operational Strengths:</b><br/>
        • <b>Institutional Infrastructure:</b> Professional-grade systems and processes<br/>
        • <b>Risk Management:</b> Dedicated risk team with comprehensive monitoring<br/>
        • <b>Diversified Team:</b> Multiple PMs reduce key person risk<br/>
        • <b>Regulatory Compliance:</b> SEC registered with proper oversight<br/><br/>
        
        <b>Operational Concerns:</b><br/>
        • <b>Employee Turnover:</b> High turnover in recent years (7 in last 2 years)<br/>
        • <b>Complex Structure:</b> Multi-PM model may create coordination challenges<br/>
        • <b>Parent Firm Dependency:</b> Shared resources create potential conflicts<br/>
        • <b>Performance Variability:</b> Individual PM performance may vary significantly<br/><br/>
        
        <b>Client Base Analysis:</b><br/>
        • Well-diversified institutional client base (60% institutional)<br/>
        • Top client represents only 10% of assets<br/>
        • Strong relationships with fund-of-funds and family offices
        """
        story.append(Paragraph(analysis_text, self.styles['Normal']))

    def _add_performance_charts_section(self, story):
        """Add performance charts section with cumulative returns, S&P drawdown, and fund-specific drawdown."""
        story.append(Paragraph("PERFORMANCE VISUALIZATION", self.styles['SectionHeader']))
        
        charts_dir = "outputs/charts"
        
        # Add cumulative returns chart (same for all funds)
        cumulative_path = os.path.join(charts_dir, "cumulative_returns.png")
        if os.path.exists(cumulative_path):
            story.append(Paragraph("<b>Cumulative Returns Comparison</b>", self.styles['SubsectionHeader']))
            story.append(Image(cumulative_path, width=7*inch, height=4.2*inch))
            story.append(Spacer(1, 0.1*inch))
            
            # Add explanation text
            explanation_text = """
            <b>Chart Analysis:</b> The cumulative returns chart shows the growth of $1 invested in each fund 
            compared to the S&P 500 benchmark over the analysis period. This visualization helps identify 
            periods of outperformance and underperformance, as well as the overall trajectory of wealth creation. 
            Steeper slopes indicate higher returns, while periods of decline show drawdowns and recovery patterns.
            """
            story.append(Paragraph(explanation_text, self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add S&P 500 drawdown chart (same for all funds)
        sp500_dd_path = os.path.join(charts_dir, "drawdown_S&P_500_w__Div.png")
        if os.path.exists(sp500_dd_path):
            story.append(Paragraph("<b>S&P 500 Benchmark Drawdown Analysis</b>", self.styles['SubsectionHeader']))
            story.append(Image(sp500_dd_path, width=7*inch, height=4.2*inch))
            story.append(Spacer(1, 0.1*inch))
            
            # Add explanation text
            explanation_text = """
            <b>Chart Analysis:</b> The S&P 500 drawdown chart illustrates the benchmark's downside risk patterns 
            over time. Drawdowns represent the percentage decline from peak values, providing insight into 
            market volatility and stress periods. This serves as a baseline for comparing individual fund 
            drawdown characteristics and risk management effectiveness.
            """
            story.append(Paragraph(explanation_text, self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add fund-specific drawdown chart
        fund_dd_filename = f"drawdown_{self.fund_name.replace(' ', '_')}.png"
        fund_dd_path = os.path.join(charts_dir, fund_dd_filename)
        if os.path.exists(fund_dd_path):
            story.append(Paragraph(f"<b>{self.fund_name} Drawdown Analysis</b>", self.styles['SubsectionHeader']))
            story.append(Image(fund_dd_path, width=7*inch, height=4.2*inch))
            story.append(Spacer(1, 0.1*inch))
            
            # Add explanation text
            explanation_text = f"""
            <b>Chart Analysis:</b> The {self.fund_name} drawdown chart reveals the fund's specific risk patterns 
            and downside protection capabilities. Comparing this to the S&P 500 benchmark drawdown shows 
            whether the fund provides better downside protection during market stress periods. 
            {'Lower' if abs(self.summary_stats.get('MaxDrawdown', 0)) < 0.15 else 'Higher'} maximum drawdown 
            of {abs(self.summary_stats.get('MaxDrawdown', 0)):.1%} indicates 
            {'superior' if abs(self.summary_stats.get('MaxDrawdown', 0)) < 0.15 else 'moderate'} 
            risk management relative to typical hedge fund standards.
            """
            story.append(Paragraph(explanation_text, self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add overall chart interpretation
        overall_text = """
        <b>Overall Chart Interpretation:</b><br/>
        These visualizations provide critical insights into the fund's risk-return profile and market behavior. 
        The cumulative returns chart demonstrates wealth creation over time, while the drawdown charts reveal 
        downside risk characteristics. Together, they complement the quantitative metrics by showing the 
        temporal patterns of performance and risk, helping investors understand not just what returns were 
        achieved, but how they were achieved and what risks were taken along the way.
        """
        story.append(Paragraph(overall_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

    def create_recommendation(self, story):
        """Create investment recommendation section."""
        story.append(Paragraph("INVESTMENT RECOMMENDATION", self.styles['SectionHeader']))
        
        # Fund-specific conclusions
        if "Fund 1" in self.fund_name:
            recommendation = "HOLD"
            rationale = "Good track record, but operational fragility and one-man dependency. Not fraudulent, but not institutionally robust."
        elif "Fund 2" in self.fund_name:
            recommendation = "AVOID"
            rationale = "This one looks sus — unrealistic risk-return asymmetry, opaque client disclosures, high fees, questionable backstory, and discretionary risk management. Possible style drift or data massaging."
        elif "Fund 3" in self.fund_name:
            recommendation = "STRONG BUY"
            rationale = "This is the most institutional and allocatable choice: strong governance, infrastructure, low correlation, consistent returns — ideal complement for BDPRF's long-only S&P portfolio."
        else:
            recommendation = "HOLD"
            rationale = "Standard hedge fund analysis required."
        
        # Get key metrics for display
        sharpe = self.summary_stats.get('Sharpe', 0)
        alpha = self.summary_stats.get('Alpha_ann', 0)
        max_dd = abs(self.summary_stats.get('MaxDrawdown', 0))
        calmar = self.summary_stats.get('Calmar', 0)
        
        recommendation_text = f"""
        <b>Recommendation: {recommendation}</b><br/><br/>
        
        <b>Rationale:</b><br/>
        {rationale}<br/><br/>
        
        <b>Key Considerations:</b><br/>
        • Sharpe Ratio: {sharpe:.2f} ({'Excellent' if sharpe > 1.5 else 'Good' if sharpe > 1.0 else 'Fair' if sharpe > 0.5 else 'Poor'})<br/>
        • Alpha Generation: {alpha:.1%} ({'Strong' if alpha > 0.05 else 'Moderate' if alpha > 0.02 else 'Weak' if alpha > 0 else 'Negative'})<br/>
        • Maximum Drawdown: {max_dd:.1%} ({'Low' if max_dd < 0.10 else 'Moderate' if max_dd < 0.20 else 'High'})<br/>
        • Calmar Ratio: {calmar:.2f} ({'Excellent' if calmar > 2.0 else 'Good' if calmar > 1.0 else 'Fair' if calmar > 0.5 else 'Poor'})<br/><br/>
        
        <b>Additional Information Needed:</b><br/>
        • Detailed operational risk assessment<br/>
        • Legal and compliance review<br/>
        • Reference checks with existing investors<br/>
        • On-site due diligence visit<br/>
        • Detailed fee structure analysis
        """
        
        story.append(Paragraph(recommendation_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))


    def generate_pdf(self, output_path):
        """Generate the complete PDF report."""
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Create all sections
        self.create_title_page(story)
        self.create_executive_summary(story)
        self.create_quantitative_analysis(story)
        self.create_investment_due_diligence(story)
        self.create_operational_due_diligence(story)
        self.create_recommendation(story)
        
        # Build PDF
        doc.build(story)

def main():
    """Main function to generate PDFs for all funds."""
    # Load data
    data_path = "data/fund_returns_clean.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date")
    
    # Load summary statistics
    summary_df = pd.read_csv("outputs/tables/summary_stats.csv", index_col=0)
    
    # Create output directory
    os.makedirs("outputs/pdfs", exist_ok=True)
    
    # Generate PDF for each fund
    funds = ["Fund 1", "Fund 2", "Fund 3"]
    
    for fund in funds:
        print(f"Generating PDF for {fund}...")
        
        # Get fund data and stats
        fund_data = df[fund]
        benchmark_data = df["S&P 500 w/ Div"]
        fund_stats = summary_df.loc[fund].to_dict()
        
        # Create PDF generator
        pdf_generator = FundAnalysisPDF(fund, fund_data, benchmark_data, fund_stats)
        
        # Generate PDF
        output_path = f"outputs/pdfs/{fund.replace(' ', '_')}_Due_Diligence_Report.pdf"
        pdf_generator.generate_pdf(output_path)
        
        print(f"✓ Generated: {output_path}")
    
    print("\nAll PDF reports generated successfully!")

if __name__ == "__main__":
    main()
