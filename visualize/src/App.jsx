import { Routes, Route } from 'react-router-dom'
import Nav from './components/Nav.jsx'
import Home from './pages/Home.jsx'
import GenderClassifier from './pages/GenderClassifier.jsx'
import AdjectiveCategories from './pages/AdjectiveCategories.jsx'
import LogOdds from './pages/LogOdds.jsx'

export default function App() {
  return (
    <div className="app">
      <Nav />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/gender-classifier" element={<GenderClassifier />} />
          <Route path="/adjective-categories" element={<AdjectiveCategories />} />
          <Route path="/log-odds" element={<LogOdds />} />
        </Routes>
      </main>
    </div>
  )
}
