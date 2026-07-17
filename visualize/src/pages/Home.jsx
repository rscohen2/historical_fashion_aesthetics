import { Link } from 'react-router-dom'

const pages = [
  {
    to: '/gender-classifier',
    title: 'Gender Classifier',
    description: 'Per-decade logistic regression coefficients showing how fashion terms correlate with character gender across time.',
  },
  {
    to: '/adjective-categories',
    title: 'Adjective Categories',
    description: 'Distribution of semantic adjective categories applied to fashion descriptions in the corpus.',
  },
  {
    to: '/log-odds',
    title: 'Character Adjective Log-Odds',
    description: 'Which character adjectives are statistically associated with each fashion term across the corpus.',
  },
  {
    to: '/unique-items',
    title: 'Unique Items',
    description: 'The most fashion-y characters, explorable as cards, with a scatter of fashion mentions against distinct items.',
  },
]

export default function Home() {
  return (
    <div>
      <div className="page-header">
        <h1>Historical Fashion Aesthetics</h1>
        <p>Interactive visualizations of fashion trends extracted from a corpus of novels.</p>
      </div>
      <div className="home-grid">
        {pages.map(({ to, title, description }) => (
          <Link key={to} to={to} className="home-card">
            <h2>{title}</h2>
            <p>{description}</p>
          </Link>
        ))}
      </div>
      <style>{`
        .home-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 1rem;
        }
        .home-card {
          display: block;
          text-decoration: none;
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 1.5rem;
          background: var(--bg);
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        .home-card:hover {
          border-color: var(--accent);
          box-shadow: 0 4px 12px rgba(79,70,229,0.1);
        }
        .home-card h2 { color: var(--accent); margin-bottom: 0.5rem; }
        .home-card p { color: var(--text-muted); font-size: 0.875rem; margin: 0; }
      `}</style>
    </div>
  )
}
