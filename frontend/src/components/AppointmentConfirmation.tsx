// /frontend/src/components/AppointmentConfirmation.tsx
import React from 'react';
import { Calendar, MapPin, Clock, Users, FileText, CheckCircle, XCircle } from 'lucide-react';

interface AppointmentDetails {
  title: string;
  location: string;
  datetime: string;
  duration: number;
  attendees: string;
  description?: string;
}

interface AppointmentConfirmationProps {
  appointment: AppointmentDetails;
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
  // Check if appointment data is complete
  const isComplete = appointment.location && appointment.datetime && appointment.duration > 0;
  
  // Format missing fields for display
  const getMissingFields = () => {
    const missing = [];
    if (!appointment.location) missing.push('Location');
    if (!appointment.datetime || appointment.datetime.includes('1970') || appointment.datetime.includes('1969')) missing.push('Date & Time');
    if (!appointment.duration || appointment.duration <= 0) missing.push('Duration');
    return missing;
  };

  const missingFields = getMissingFields();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              <Calendar className="w-5 h-5 text-blue-600" />
              {isComplete ? 'Appointment Confirmation' : 'Appointment Details'}
            </h2>
            <button
              onClick={onCancel}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <XCircle className="w-6 h-6" />
            </button>
          </div>

          <div className="space-y-4">
            {/* Title */}
            <div className="flex items-start gap-3">
              <FileText className="w-5 h-5 text-gray-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Title</p>
                <p className="text-gray-900 dark:text-white">{appointment.title || 'Meeting'}</p>
              </div>
            </div>

            {/* Location */}
            <div className="flex items-start gap-3">
              <MapPin className="w-5 h-5 text-gray-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Location</p>
                {appointment.location ? (
                  <p className="text-gray-900 dark:text-white">{appointment.location}</p>
                ) : (
                  <p className="text-orange-600 dark:text-orange-400 text-sm italic">Location needed</p>
                )}
              </div>
            </div>

            {/* Date & Time */}
            <div className="flex items-start gap-3">
              <Calendar className="w-5 h-5 text-gray-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Date & Time</p>
                {appointment.datetime && !appointment.datetime.includes('1970') && !appointment.datetime.includes('1969') ? (
                  <p className="text-gray-900 dark:text-white">{appointment.datetime}</p>
                ) : (
                  <p className="text-orange-600 dark:text-orange-400 text-sm italic">Date & time needed</p>
                )}
              </div>
            </div>

            {/* Duration */}
            <div className="flex items-start gap-3">
              <Clock className="w-5 h-5 text-gray-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Duration</p>
                {appointment.duration && appointment.duration > 0 ? (
                  <p className="text-gray-900 dark:text-white">{appointment.duration} minutes</p>
                ) : (
                  <p className="text-orange-600 dark:text-orange-400 text-sm italic">Duration needed</p>
                )}
              </div>
            </div>

            {/* Attendees */}
            {appointment.attendees && (
              <div className="flex items-start gap-3">
                <Users className="w-5 h-5 text-gray-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Attendees</p>
                  <p className="text-gray-900 dark:text-white">{appointment.attendees}</p>
                </div>
              </div>
            )}

            {/* Description */}
            {appointment.description && (
              <div className="flex items-start gap-3">
                <FileText className="w-5 h-5 text-gray-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Description</p>
                  <p className="text-gray-900 dark:text-white">{appointment.description}</p>
                </div>
              </div>
            )}
          </div>

          {/* Missing fields warning */}
          {!isComplete && missingFields.length > 0 && (
            <div className="mt-4 p-3 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg">
              <p className="text-sm text-orange-800 dark:text-orange-200">
                <strong>Please provide:</strong> {missingFields.join(', ')}
              </p>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-3 mt-6">
            <button
              onClick={onCancel}
              disabled={isLoading}
              className="flex-1 px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              disabled={isLoading || !isComplete}
              className="flex-1 px-4 py-2 text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Confirming...
                </>
              ) : (
                <>
                  <CheckCircle className="w-4 h-4" />
                  {isComplete ? 'Confirm Appointment' : 'Complete Details First'}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AppointmentConfirmation; 