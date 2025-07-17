// /frontend/src/components/AppointmentCard.tsx
import React from 'react';

interface AppointmentCardProps {
  title: string;
  location: string;
  datetime: string;
  duration: number;
  organizer?: string;
  meetLink?: string;
  status?: 'upcoming' | 'confirmed' | 'cancelled';
  onClick?: () => void;
}

const AppointmentCard: React.FC<AppointmentCardProps> = ({
  title,
  location,
  datetime,
  duration,
  organizer,
  meetLink,
  status = 'confirmed',
  onClick
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'upcoming':
        return '⏳';
      case 'confirmed':
        return '✅';
      case 'cancelled':
        return '❌';
      default:
        return '📅';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'upcoming':
        return '#f59e0b';
      case 'confirmed':
        return '#10b981';
      case 'cancelled':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  return (
    <div className={`appointment-card ${onClick ? 'clickable' : ''}`} onClick={onClick}>
      <div className="card-content">
        <div className="card-header">
          <div className="status-indicator" style={{ color: getStatusColor() }}>
            {getStatusIcon()}
          </div>
          <h3 className="appointment-title">{title}</h3>
        </div>

        <div className="appointment-info">
          <div className="info-row">
            <span className="info-icon">📍</span>
            <span className="info-text">{location}</span>
          </div>

          <div className="info-row">
            <span className="info-icon">🕐</span>
            <span className="info-text">{datetime}</span>
          </div>

          <div className="info-row">
            <span className="info-icon">⏱️</span>
            <span className="info-text">{duration} minutes</span>
          </div>

          {organizer && (
            <div className="info-row">
              <span className="info-icon">👤</span>
              <span className="info-text">{organizer}</span>
            </div>
          )}
        </div>

        {meetLink && (
          <div className="meet-link-section">
            <a 
              href={meetLink} 
              target="_blank" 
              rel="noopener noreferrer"
              className="meet-link"
              onClick={(e) => e.stopPropagation()}
            >
              🎥 Join Google Meet
            </a>
          </div>
        )}
      </div>

      <style jsx>{`
        .appointment-card {
          background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 16px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          transition: all 0.2s ease;
          position: relative;
          overflow: hidden;
        }

        .appointment-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 4px;
          background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }

        .appointment-card.clickable {
          cursor: pointer;
        }

        .appointment-card.clickable:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.15);
          border-color: #c7d2fe;
        }

        .card-content {
          position: relative;
          z-index: 1;
        }

        .card-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 16px;
        }

        .status-indicator {
          font-size: 24px;
          line-height: 1;
        }

        .appointment-title {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
          color: #1f2937;
          flex: 1;
        }

        .appointment-info {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 16px;
        }

        .info-row {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .info-icon {
          font-size: 16px;
          min-width: 20px;
          text-align: center;
        }

        .info-text {
          color: #4b5563;
          font-size: 14px;
          font-weight: 500;
        }

        .meet-link-section {
          padding-top: 12px;
          border-top: 1px solid #e5e7eb;
        }

        .meet-link {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
          color: white;
          text-decoration: none;
          padding: 8px 16px;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 600;
          transition: all 0.2s ease;
        }

        .meet-link:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        @media (max-width: 640px) {
          .appointment-card {
            padding: 16px;
          }

          .appointment-title {
            font-size: 16px;
          }

          .info-text {
            font-size: 13px;
          }
        }
      `}</style>
    </div>
  );
};

export default AppointmentCard; 