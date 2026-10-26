
import { Routes, Route } from "react-router-dom";

import Home from "./pages/Home";
import JobProgress from "./pages/JobProgress";
import JobHistory from "./pages/JobHistory";

import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/job/:id" element={<JobProgress />} />
          <Route path="/history" element={<JobHistory />} />
        </Routes>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;