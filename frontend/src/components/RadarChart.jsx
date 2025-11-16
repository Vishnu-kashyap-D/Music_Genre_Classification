import { Radar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
)

export default function RadarChart({ data }) {
  const genres = Object.keys(data)
  const values = Object.values(data)

  const chartData = {
    labels: genres.map(g => g.charAt(0).toUpperCase() + g.slice(1)),
    datasets: [
      {
        label: 'Confidence',
        data: values,
        backgroundColor: 'rgba(168, 85, 247, 0.2)',
        borderColor: 'rgba(168, 85, 247, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(168, 85, 247, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(168, 85, 247, 1)',
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    scales: {
      r: {
        beginAtZero: true,
        max: 1,
        ticks: {
          stepSize: 0.2,
          color: '#a0a0a0',
        },
        grid: {
          color: '#333333',
        },
        pointLabels: {
          color: '#ffffff',
          font: {
            size: 12,
            weight: 'bold',
          },
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            return `${context.label}: ${(context.parsed.r * 100).toFixed(1)}%`
          },
        },
        backgroundColor: '#242424',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#a855f7',
        borderWidth: 1,
      },
    },
  }

  return (
    <div className="h-64">
      <Radar data={chartData} options={options} />
    </div>
  )
}

