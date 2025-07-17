// /frontend/src/components/AppointmentConfirmation.tsx
import React, { useState, useEffect } from 'react';

interface AppointmentData {
  title: string;
  location: string;
  datetime: string;
  duration: number;
  attendees: string;
  description?: string;
}

interface AppointmentConfirmationProps {
  appointment: AppointmentData;
  onConfirm: () => void;
  onCancel: () => void;
  isLoading?: boolean;
}

const AppointmentConfirmation: React.FC<AppointmentConfirmationProps> = ({
  appointment,
  onConfirm,
  onCancel,
  isLoading = false
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [confirmingStatus, setConfirmingStatus] = useState<'idle' | 'confirming' | 'confirmed'>('idle');

  useEffect(() => {
    // Trigger entrance animation
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const handleConfirm = async () => {
    setConfirmingStatus('confirming');
    try {
      await onConfirm();
      setConfirmingStatus('confirmed');
      // Let the success state show for a moment before hiding
      setTimeout(() => setIsVisible(false), 2000);
    } catch (error) {
      setConfirmingStatus('idle');
    }
  };

  const handleCancel = () => {
    setIsVisible(false);
    setTimeout(onCancel, 300); // Wait for exit animation
  };

  if (confirmingStatus === 'confirmed') {
    return (
      <div className={`appointment-confirmation ${isVisible ? 'visible' : ''}`}>
        <div className="appointment-card success-card">
          <div className="success-animation">
            <div className="checkmark">✅</div>
          </div>
          <h2>Appointment Confirmed! 🎉</h2>
          <p>You'll receive a calendar invitation with Google Meet link shortly.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`appointment-confirmation ${isVisible ? 'visible' : ''}`}>
      <div className="appointment-card">
        <div className="card-header">
          <h2>📅 Appointment Confirmation</h2>
          <p>Please review and confirm your appointment details</p>
        </div>

        <div className="appointment-details">
          <div className="detail-row">
            <div className="detail-icon">📋</div>
            <div className="detail-content">
              <strong>Meeting</strong>
              <span>{appointment.title}</span>
            </div>
          </div>

          <div className="detail-row">
            <div className="detail-icon">📍</div>
            <div className="detail-content">
              <strong>Location</strong>
              <span>{appointment.location}</span>
            </div>
          </div>

          <div className="detail-row">
            <div className="detail-icon">🕐</div>
            <div className="detail-content">
              <strong>Date & Time</strong>
              <span>{appointment.datetime}</span>
            </div>
          </div>

          <div className="detail-row">
            <div className="detail-icon">⏱️</div>
            <div className="detail-content">
              <strong>Duration</strong>
              <span>{appointment.duration} minutes</span>
            </div>
          </div>

          <div className="detail-row">
            <div className="detail-icon">👥</div>
            <div className="detail-content">
              <strong>Attendees</strong>
              <span>{appointment.attendees}</span>
            </div>
          </div>

          {appointment.description && (
            <div className="detail-row">
              <div className="detail-icon">📝</div>
              <div className="detail-content">
                <strong>Description</strong>
                <span>{appointment.description}</span>
              </div>
            </div>
          )}
        </div>

        <div className="meet-info">
          <div className="meet-icon">🎥</div>
          <div className="meet-text">
            <strong>Google Meet Included</strong>
            <span>A video conference link will be automatically created</span>
          </div>
        </div>

        <div className="action-buttons">
          <button
            className={`confirm-button ${confirmingStatus === 'confirming' ? 'loading' : ''}`}
            onClick={handleConfirm}
            disabled={isLoading || confirmingStatus === 'confirming'}
          >
            {confirmingStatus === 'confirming' ? (
              <>
                <span className="spinner"></span>
                Confirming...
              </>
            ) : (
              <>
                ✅ Confirm Appointment
              </>
            )}
          </button>

          <button
            className="cancel-button"
            onClick={handleCancel}
            disabled={isLoading || confirmingStatus === 'confirming'}
          >
            ❌ Cancel
          </button>
        </div>
      </div>

      <style jsx>{`
        .appointment-confirmation {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          opacity: 0;
          transform: translateY(20px);
          transition: all 0.3s ease-out;
        }

        .appointment-confirmation.visible {
          opacity: 1;
          transform: translateY(0);
        }

        .appointment-card {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 16px;
          padding: 32px;
          max-width: 500px;
          width: 90%;
          max-height: 90vh;
          overflow-y: auto;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
          color: white;
          position: relative;
          transform: scale(0.9);
          transition: transform 0.3s ease-out;
        }

        .appointment-confirmation.visible .appointment-card {
          transform: scale(1);
        }

        .success-card {
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          text-align: center;
        }

        .card-header {
          text-align: center;
          margin-bottom: 32px;
        }

        .card-header h2 {
          margin: 0 0 8px 0;
          font-size: 24px;
          font-weight: 700;
        }

        .card-header p {
          margin: 0;
          opacity: 0.9;
          font-size: 16px;
        }

        .appointment-details {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 24px;
          margin-bottom: 24px;
          backdrop-filter: blur(10px);
        }

        .detail-row {
          display: flex;
          align-items: flex-start;
          margin-bottom: 16px;
          gap: 12px;
        }

        .detail-row:last-child {
          margin-bottom: 0;
        }

        .detail-icon {
          font-size: 20px;
          min-width: 24px;
          text-align: center;
        }

        .detail-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .detail-content strong {
          font-weight: 600;
          font-size: 14px;
          opacity: 0.8;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .detail-content span {
          font-size: 16px;
          font-weight: 500;
        }

        .meet-info {
          background: rgba(59, 130, 246, 0.2);
          border: 1px solid rgba(59, 130, 246, 0.3);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 24px;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .meet-icon {
          font-size: 24px;
        }

        .meet-text {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .meet-text strong {
          font-weight: 600;
          font-size: 16px;
        }

        .meet-text span {
          font-size: 14px;
          opacity: 0.9;
        }

        .action-buttons {
          display: flex;
          gap: 16px;
          flex-direction: column;
        }

        .confirm-button, .cancel-button {
          padding: 16px 24px;
          border: none;
          border-radius: 8px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          position: relative;
        }

        .confirm-button {
          background: #10b981;
          color: white;
        }

        .confirm-button:hover:not(:disabled) {
          background: #059669;
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        }

        .confirm-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
          transform: none;
        }

        .cancel-button {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .cancel-button:hover:not(:disabled) {
          background: rgba(239, 68, 68, 0.2);
          border-color: rgba(239, 68, 68, 0.5);
          transform: translateY(-2px);
        }

        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top: 2px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        .success-animation {
          margin-bottom: 24px;
        }

        .checkmark {
          font-size: 64px;
          animation: bounceIn 0.6s ease-out;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @keyframes bounceIn {
          0% {
            transform: scale(0.3);
            opacity: 0;
          }
          50% {
            transform: scale(1.05);
          }
          70% {
            transform: scale(0.9);
          }
          100% {
            transform: scale(1);
            opacity: 1;
          }
        }

        @media (min-width: 640px) {
          .action-buttons {
            flex-direction: row;
          }
        }
      `}</style>
    </div>
  );
};

export default AppointmentConfirmation; 