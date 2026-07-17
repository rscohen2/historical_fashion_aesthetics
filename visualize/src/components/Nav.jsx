import { NavLink } from 'react-router-dom'
import './Nav.css'

const pages = [
  { to: '/', label: 'Home' },
  { to: '/gender-classifier', label: 'Gender Classifier' },
  { to: '/adjective-categories', label: 'Adjective Categories' },
  { to: '/log-odds', label: 'Log-Odds' },
  { to: '/unique-items', label: 'Unique Items' },
]

export default function Nav() {
  return (
    <nav className="nav">
      <span className="nav-title">Fashion Visualizations</span>
      <ul>
        {pages.map(({ to, label }) => (
          <li key={to}>
            <NavLink to={to} end={to === '/'} className={({ isActive }) => isActive ? 'active' : ''}>
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  )
}
