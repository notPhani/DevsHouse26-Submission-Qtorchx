import React from 'react';
import './RealWorldApplications.css';

const RealWorldApplications = () => {
  // ✅ EXACTLY 6 STEPS - HORIZONTAL PIPELINE (LEFT → RIGHT)
  const pipelineSteps = [
    {
      id: 1,
      label: 'STEP 01',
      title: 'Quantum Hardware Optimization',
      description: 'Noise originates and accumulates in physical qubits through decoherence.',
      color: '#3498db'
    },
    {
      id: 2,
      label: 'STEP 02',
      title: 'Noise Analysis',
      description: 'Analyze how noise patterns evolve across circuit depth.',
      color: '#9b59b6'
    },
    {
      id: 3,
      label: 'STEP 03',
      title: 'Error Mitigation & Correction',
      description: 'Predict error propagation and apply targeted mitigation.',
      color: '#2ecc71'
    },
    {
      id: 4,
      label: 'STEP 04',
      title: 'Realistic Simulation',
      description: 'Use corrected noise models to simulate real device behavior.',
      color: '#e74c3c'
    },
    {
      id: 5,
      label: 'STEP 05',
      title: 'Noise Pattern Learning',
      description: 'Learn and capture device-specific noise fingerprints.',
      color: '#f39c12'
    },
    {
      id: 6,
      label: 'STEP 06',
      title: 'Scientific Research & Insights',
      description: 'Extract deep quantum insights and advance physics understanding.',
      color: '#d35400'
    }
  ];

  // ✅ EXACTLY 6 APPLICATIONS - GRID LAYOUT
  const applications = [
    {
      id: 1,
      category: 'BIOTECH',
      title: 'Drug Discovery & Molecular Simulation',
      description: 'Simulate molecular binding with noise-aware quantum algorithms to accelerate drug candidate screening and molecular design.',
      color: '#3498db'
    },
    {
      id: 2,
      category: 'SECURITY',
      title: 'Cryptography & Cybersecurity',
      description: 'Build noise-resilient quantum cryptographic systems and prepare for post-quantum security with accurate threat modeling.',
      color: '#9b59b6'
    },
    {
      id: 3,
      category: 'FINANCE',
      title: 'Finance & Optimization',
      description: 'Optimize portfolios and risk models with quantum algorithms that account for real hardware noise and error rates.',
      color: '#2ecc71'
    },
    {
      id: 4,
      category: 'LOGISTICS',
      title: 'Logistics & Route Optimization',
      description: 'Solve complex combinatorial problems with QAOA and VQE algorithms designed specifically for noisy quantum devices.',
      color: '#f39c12'
    },
    {
      id: 5,
      category: 'MATERIALS',
      title: 'Material Science & Energy Systems',
      description: 'Discover new materials and battery chemistries by simulating quantum properties on hardware-aware quantum computers.',
      color: '#e74c3c'
    },
    {
      id: 6,
      category: 'AI',
      title: 'AI + Quantum Hybrid Systems',
      description: 'Combine quantum and classical AI workflows with noise-safe quantum subroutines for enhanced machine learning performance.',
      color: '#d35400'
    }
  ];

  return (
    <div className="rwa-page">
      {/* ═━═━═━═━━═━═━ HERO SECTION ═━═━═━═━━═━ */}
      <section className="rwa-hero">
        <div className="rwa-container">
          <h1 className="rwa-hero-title">
            Real-World Applications of <span className="rwa-accent">QNaF</span>
          </h1>
          <p className="rwa-hero-subtitle">
            QNaF models quantum noise as a dynamic field evolving across space and time, enabling realistic simulation, optimization, and better quantum system design.
          </p>
        </div>
      </section>

      {/* ═━═━═━═━━═━═━ SECTION 1: HORIZONTAL PIPELINE (6 STEPS LEFT→RIGHT) ═━═━━ */}
      <section className="rwa-section rwa-pipeline-section">
        <div className="rwa-container">
          <div className="rwa-section-header">
            <h2 className="rwa-section-title">
              Understanding Quantum Noise <span className="rwa-accent">Step by Step</span>
            </h2>
            <p className="rwa-section-subtitle">
              QNaF's modular approach breaks down quantum noise analysis into a logical, step-by-step workflow.
            </p>
          </div>

          {/* HORIZONTAL PIPELINE - ALL 6 STEPS VISIBLE */}
          <div className="rwa-horizontal-pipeline">
            {pipelineSteps.map((step, index) => (
              <React.Fragment key={step.id}>
                {/* STEP CARD */}
                <div 
                  className="rwa-pipeline-step"
                  style={{ '--step-color': step.color }}
                >
                  <div className="rwa-pipeline-step-label">{step.label}</div>
                  <h3 className="rwa-pipeline-step-title">{step.title}</h3>
                  <p className="rwa-pipeline-step-desc">{step.description}</p>
                </div>

                {/* ARROW BETWEEN STEPS (except after last) */}
                {index < pipelineSteps.length - 1 && (
                  <div className="rwa-pipeline-arrow" style={{ '--arrow-color': step.color }}>
                    <span>→</span>
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </section>

      {/* ═━═━═━═━━═━═━ SECTION 2: REAL-WORLD APPLICATIONS GRID (6 CARDS) ═━═━ */}
      <section className="rwa-section rwa-applications-section">
        <div className="rwa-container">
          <div className="rwa-section-header">
            <h2 className="rwa-section-title">
              Quantum Solutions <span className="rwa-accent">Across Industries</span>
            </h2>
            <p className="rwa-section-subtitle">
              QNaF unlocks practical quantum computing across critical domains.
            </p>
          </div>

          {/* 3-COLUMN RESPONSIVE GRID */}
          <div className="rwa-applications-grid">
            {applications.map((app) => (
              <div 
                key={app.id}
                className="rwa-app-card"
                style={{ '--app-color': app.color }}
              >
                <span className="rwa-app-category">{app.category}</span>
                <h3 className="rwa-app-card-title">{app.title}</h3>
                <p className="rwa-app-card-desc">{app.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default RealWorldApplications;
