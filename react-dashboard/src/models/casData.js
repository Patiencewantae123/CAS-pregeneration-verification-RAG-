export const BASELINE_METRICS = [
  { name: 'Baseline RAG', accuracy: 72.1, hallucination: 21.4, contradiction: 18.9, consistency: 70.3, cpi: 75.48 },
  { name: 'Self-RAG', accuracy: 76.8, hallucination: 16.2, contradiction: 14.1, consistency: 77.5, cpi: 81.00 },
  { name: 'CRAG', accuracy: 78.4, hallucination: 14.8, contradiction: 12.9, consistency: 79.2, cpi: 82.48 },
  { name: 'CAS (Proposed)', accuracy: 82.7, hallucination: 9.5, contradiction: 6.1, consistency: 86.8, cpi: 88.48 },
];

export const INITIAL_DOCUMENTS = [
  { id: 'Doc_01', source: 'PeerReviewed_Journal', content: 'The FDA approved drug-X for Phase 3 clinical trials in early 2024.', rel: 0.90, prov: 0.95, conf: 0.92 },
  { id: 'Doc_02', source: 'Unverified_Blog', content: 'Regulators rejected drug-X for Phase 3 trials due to severe health concerns.', rel: 0.75, prov: 0.25, conf: 0.65 },
  { id: 'Doc_03', source: 'Medical_News_Outlet', content: 'Phase 3 clinical trial for drug-X was approved by health regulatory bodies.', rel: 0.85, prov: 0.80, conf: 0.88 },
  { id: 'Doc_04', source: 'Pharma_DB', content: 'Drug-X is an experimental therapeutic compound tested for hypertension.', rel: 0.92, prov: 0.90, conf: 0.95 }
];
