//! Compliance and tax reporting exports
//!
//! This module provides functionality for generating compliance-ready exports:
//! - Tax reports (capital gains, cost basis)
//! - Audit trails with HMAC verification
//! - Multiple export formats (CSV, JSON)
//! - IRS Form 8949 compatible data

use chrono::{DateTime, Utc, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::error::{Error, Result};
use crate::models::{Holding, Portfolio, AuditLogEntry};

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Json,
    TaxForm8949,
}

/// Tax lot for cost basis tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxLot {
    pub holding_id: String,
    pub ticker: String,
    pub quantity: f64,
    pub cost_basis_per_share: f64,
    pub total_cost: f64,
    pub acquisition_date: DateTime<Utc>,
    pub holding_period: HoldingPeriod,
}

/// Holding period classification for tax purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HoldingPeriod {
    /// Less than 1 year - short-term capital gains
    ShortTerm,
    /// 1 year or more - long-term capital gains
    LongTerm,
}

impl HoldingPeriod {
    pub fn from_acquisition_date(acquisition_date: DateTime<Utc>) -> Self {
        let now = Utc::now();
        let days_held = (now - acquisition_date).num_days();
        if days_held >= 365 {
            HoldingPeriod::LongTerm
        } else {
            HoldingPeriod::ShortTerm
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            HoldingPeriod::ShortTerm => "Short-term",
            HoldingPeriod::LongTerm => "Long-term",
        }
    }
}

/// Capital gain/loss record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapitalGainRecord {
    pub ticker: String,
    pub description: String,
    pub quantity: f64,
    pub acquisition_date: String,
    pub sale_date: String,
    pub proceeds: f64,
    pub cost_basis: f64,
    pub gain_or_loss: f64,
    pub holding_period: HoldingPeriod,
    /// IRS Form 8949 box code
    pub form_8949_box: Form8949Box,
}

/// IRS Form 8949 box codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Form8949Box {
    /// Box A: Short-term, basis reported to IRS
    BoxA,
    /// Box B: Short-term, basis not reported to IRS
    BoxB,
    /// Box C: Short-term, no 1099-B
    BoxC,
    /// Box D: Long-term, basis reported to IRS
    BoxD,
    /// Box E: Long-term, basis not reported to IRS
    BoxE,
    /// Box F: Long-term, no 1099-B
    BoxF,
}

impl Form8949Box {
    pub fn as_str(&self) -> &'static str {
        match self {
            Form8949Box::BoxA => "A",
            Form8949Box::BoxB => "B",
            Form8949Box::BoxC => "C",
            Form8949Box::BoxD => "D",
            Form8949Box::BoxE => "E",
            Form8949Box::BoxF => "F",
        }
    }
}

/// Portfolio summary for tax reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxSummary {
    pub tax_year: i32,
    pub total_proceeds: f64,
    pub total_cost_basis: f64,
    pub total_gain_or_loss: f64,
    pub short_term_gain: f64,
    pub short_term_loss: f64,
    pub long_term_gain: f64,
    pub long_term_loss: f64,
    pub wash_sale_adjustments: f64,
    pub net_short_term: f64,
    pub net_long_term: f64,
}

/// Audit trail export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditExport {
    pub entries: Vec<AuditExportEntry>,
    pub export_date: DateTime<Utc>,
    pub export_hash: String,
    pub verified: bool,
}

/// Single audit entry for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditExportEntry {
    pub timestamp: String,
    pub action: String,
    pub entity_type: String,
    pub entity_id: Option<String>,
    pub details: Option<String>,
    pub integrity_verified: bool,
}

/// Compliance exporter
pub struct ComplianceExporter<'a> {
    db: &'a Database,
}

impl<'a> ComplianceExporter<'a> {
    /// Create a new compliance exporter
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Export holdings to CSV format
    pub fn export_holdings_csv(&self, portfolio_id: Option<&str>) -> Result<String> {
        let holdings = if let Some(pid) = portfolio_id {
            self.db.list_holdings(pid)?
        } else {
            self.db.list_all_holdings()?
        };

        let mut csv = String::new();
        csv.push_str("Portfolio ID,Ticker,Name,Quantity,Cost Basis,Total Cost,Acquisition Date,Asset Class,Holding Period\n");

        for holding in holdings {
            let total_cost = holding.quantity * holding.cost_basis;
            let holding_period = HoldingPeriod::from_acquisition_date(holding.acquisition_date);

            csv.push_str(&format!(
                "{},{},{},{},{:.2},{:.2},{},{},{}\n",
                holding.portfolio_id,
                holding.ticker,
                holding.name.as_deref().unwrap_or(""),
                holding.quantity,
                holding.cost_basis,
                total_cost,
                holding.acquisition_date.format("%Y-%m-%d"),
                holding.asset_class.as_str(),
                holding_period.as_str(),
            ));
        }

        Ok(csv)
    }

    /// Export holdings to JSON format
    pub fn export_holdings_json(&self, portfolio_id: Option<&str>) -> Result<String> {
        let holdings = if let Some(pid) = portfolio_id {
            self.db.list_holdings(pid)?
        } else {
            self.db.list_all_holdings()?
        };

        let tax_lots: Vec<TaxLot> = holdings
            .into_iter()
            .map(|h| TaxLot {
                holding_id: h.id,
                ticker: h.ticker,
                quantity: h.quantity,
                cost_basis_per_share: h.cost_basis,
                total_cost: h.quantity * h.cost_basis,
                acquisition_date: h.acquisition_date,
                holding_period: HoldingPeriod::from_acquisition_date(h.acquisition_date),
            })
            .collect();

        serde_json::to_string_pretty(&tax_lots).map_err(|e| Error::SerializationError(e.to_string()))
    }

    /// Generate tax lot report
    pub fn generate_tax_lots(&self, portfolio_id: Option<&str>) -> Result<Vec<TaxLot>> {
        let holdings = if let Some(pid) = portfolio_id {
            self.db.list_holdings(pid)?
        } else {
            self.db.list_all_holdings()?
        };

        let tax_lots: Vec<TaxLot> = holdings
            .into_iter()
            .map(|h| TaxLot {
                holding_id: h.id,
                ticker: h.ticker,
                quantity: h.quantity,
                cost_basis_per_share: h.cost_basis,
                total_cost: h.quantity * h.cost_basis,
                acquisition_date: h.acquisition_date,
                holding_period: HoldingPeriod::from_acquisition_date(h.acquisition_date),
            })
            .collect();

        Ok(tax_lots)
    }

    /// Generate Form 8949 compatible CSV
    pub fn export_form_8949_csv(
        &self,
        capital_gains: &[CapitalGainRecord],
    ) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("Description,Date Acquired,Date Sold,Proceeds,Cost Basis,Adjustment,Gain or Loss,Box\n");

        for record in capital_gains {
            csv.push_str(&format!(
                "{} - {} shares,{},{},{:.2},{:.2},0,{:.2},{}\n",
                record.ticker,
                record.quantity,
                record.acquisition_date,
                record.sale_date,
                record.proceeds,
                record.cost_basis,
                record.gain_or_loss,
                record.form_8949_box.as_str(),
            ));
        }

        Ok(csv)
    }

    /// Export audit trail with integrity verification
    pub fn export_audit_trail(
        &self,
        start_date: Option<DateTime<Utc>>,
        end_date: Option<DateTime<Utc>>,
    ) -> Result<AuditExport> {
        // Verify audit log integrity first
        let verified = self.db.verify_audit_log()?;

        // Get audit entries (would need to add a method to db.rs for this)
        let entries: Vec<AuditExportEntry> = Vec::new(); // Placeholder

        // Generate export hash
        let export_date = Utc::now();
        let export_data = serde_json::to_string(&entries)
            .map_err(|e| Error::SerializationError(e.to_string()))?;
        let export_hash = hex::encode(crate::crypto::sha256(export_data.as_bytes()));

        Ok(AuditExport {
            entries,
            export_date,
            export_hash,
            verified,
        })
    }

    /// Generate portfolio cost basis report
    pub fn generate_cost_basis_report(&self, portfolio_id: Option<&str>) -> Result<CostBasisReport> {
        let holdings = if let Some(pid) = portfolio_id {
            self.db.list_holdings(pid)?
        } else {
            self.db.list_all_holdings()?
        };

        let mut total_cost_basis = 0.0;
        let mut short_term_cost = 0.0;
        let mut long_term_cost = 0.0;

        let positions: Vec<CostBasisPosition> = holdings
            .into_iter()
            .map(|h| {
                let cost = h.quantity * h.cost_basis;
                total_cost_basis += cost;

                let holding_period = HoldingPeriod::from_acquisition_date(h.acquisition_date);
                match holding_period {
                    HoldingPeriod::ShortTerm => short_term_cost += cost,
                    HoldingPeriod::LongTerm => long_term_cost += cost,
                }

                CostBasisPosition {
                    ticker: h.ticker,
                    quantity: h.quantity,
                    cost_per_share: h.cost_basis,
                    total_cost: cost,
                    acquisition_date: h.acquisition_date,
                    holding_period,
                }
            })
            .collect();

        Ok(CostBasisReport {
            generated_at: Utc::now(),
            total_cost_basis,
            short_term_cost,
            long_term_cost,
            positions,
        })
    }
}

/// Cost basis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBasisReport {
    pub generated_at: DateTime<Utc>,
    pub total_cost_basis: f64,
    pub short_term_cost: f64,
    pub long_term_cost: f64,
    pub positions: Vec<CostBasisPosition>,
}

/// Single position in cost basis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBasisPosition {
    pub ticker: String,
    pub quantity: f64,
    pub cost_per_share: f64,
    pub total_cost: f64,
    pub acquisition_date: DateTime<Utc>,
    pub holding_period: HoldingPeriod,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CreatePortfolio, CreateHolding, AssetClass, PortfolioType};

    fn test_db() -> Database {
        let key = [0u8; 32];
        Database::open_in_memory(&key).unwrap()
    }

    #[test]
    fn test_holding_period_classification() {
        let now = Utc::now();

        // Short-term: acquired 6 months ago
        let short_term = now - chrono::Duration::days(180);
        assert_eq!(
            HoldingPeriod::from_acquisition_date(short_term),
            HoldingPeriod::ShortTerm
        );

        // Long-term: acquired 2 years ago
        let long_term = now - chrono::Duration::days(730);
        assert_eq!(
            HoldingPeriod::from_acquisition_date(long_term),
            HoldingPeriod::LongTerm
        );
    }

    #[test]
    fn test_export_holdings_csv() {
        let db = test_db();
        let portfolio = db.get_default_portfolio().unwrap();

        db.create_holding(CreateHolding {
            portfolio_id: portfolio.id.clone(),
            ticker: "AAPL".into(),
            quantity: 100.0,
            cost_basis: 150.0,
            acquisition_date: None,
            notes: None,
            asset_class: Some(AssetClass::Stock),
        })
        .unwrap();

        let exporter = ComplianceExporter::new(&db);
        let csv = exporter.export_holdings_csv(Some(&portfolio.id)).unwrap();

        assert!(csv.contains("AAPL"));
        assert!(csv.contains("100"));
        assert!(csv.contains("150"));
    }

    #[test]
    fn test_generate_cost_basis_report() {
        let db = test_db();
        let portfolio = db.get_default_portfolio().unwrap();

        db.create_holding(CreateHolding {
            portfolio_id: portfolio.id.clone(),
            ticker: "AAPL".into(),
            quantity: 10.0,
            cost_basis: 150.0,
            acquisition_date: None,
            notes: None,
            asset_class: None,
        })
        .unwrap();

        db.create_holding(CreateHolding {
            portfolio_id: portfolio.id.clone(),
            ticker: "MSFT".into(),
            quantity: 5.0,
            cost_basis: 300.0,
            acquisition_date: None,
            notes: None,
            asset_class: None,
        })
        .unwrap();

        let exporter = ComplianceExporter::new(&db);
        let report = exporter.generate_cost_basis_report(Some(&portfolio.id)).unwrap();

        assert_eq!(report.positions.len(), 2);
        // 10 * 150 + 5 * 300 = 1500 + 1500 = 3000
        assert_eq!(report.total_cost_basis, 3000.0);
    }

    #[test]
    fn test_export_form_8949() {
        let records = vec![
            CapitalGainRecord {
                ticker: "AAPL".into(),
                description: "Apple Inc.".into(),
                quantity: 10.0,
                acquisition_date: "2023-01-15".into(),
                sale_date: "2024-06-20".into(),
                proceeds: 1800.0,
                cost_basis: 1500.0,
                gain_or_loss: 300.0,
                holding_period: HoldingPeriod::LongTerm,
                form_8949_box: Form8949Box::BoxD,
            },
        ];

        let db = test_db();
        let exporter = ComplianceExporter::new(&db);
        let csv = exporter.export_form_8949_csv(&records).unwrap();

        assert!(csv.contains("AAPL"));
        assert!(csv.contains("1800.00"));
        assert!(csv.contains("1500.00"));
        assert!(csv.contains("300.00"));
        assert!(csv.contains("D"));
    }
}
