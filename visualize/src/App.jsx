import { Routes, Route } from 'react-router-dom'
import Nav from './components/Nav.jsx'
import Home from './pages/Home.jsx'
import GenderClassifier from './pages/GenderClassifier.jsx'
import AdjectiveCategories from './pages/AdjectiveCategories.jsx'
import LogOdds from './pages/LogOdds.jsx'
import UniqueItems from './pages/UniqueItems.jsx'
import AdjectiveTrends from './pages/AdjectiveTrends.jsx'
import SaeLogOdds from './pages/SaeLogOdds.jsx'
import SaeCodeTrends from './pages/SaeCodeTrends.jsx'

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
          <Route path="/unique-items" element={<UniqueItems />} />
          <Route path="/adjective-trends" element={<AdjectiveTrends />} />
          <Route path="/sae-log-odds" element={<SaeLogOdds />} />
          <Route path="/sae-code-trends" element={<SaeCodeTrends />} />
        </Routes>
      </main>
    </div>
  )
}
