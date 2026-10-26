import { useState } from "react";
import { Loader2, Upload, Video, Image as ImageIcon, ExternalLink } from "lucide-react";
import api from "../services/api";

function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [photoFile, setPhotoFile] = useState<File | null>(null);

  const [outputVideo, setOutputVideo] = useState<string>("");
  const [personFound, setPersonFound] = useState<boolean | null>(null);
  const [matchCount, setMatchCount] = useState<number>(0);
  const [timestamps, setTimestamps] = useState<string[]>([]);

  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>("");

  const handleVideoChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target.files && event.target.files.length > 0) {
      setVideoFile(event.target.files[0]);
      setOutputVideo("");
      setPersonFound(null);
      setMatchCount(0);
      setTimestamps([]);
      setError("");
    }
  };

  const handlePhotoChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target.files && event.target.files.length > 0) {
      setPhotoFile(event.target.files[0]);
      setOutputVideo("");
      setPersonFound(null);
      setMatchCount(0);
      setTimestamps([]);
      setError("");
    }
  };

  const handleSubmit = async () => {
    if (isProcessing) return;

    if (!videoFile || !photoFile) {
      setError("Please select both a video and a reference photo.");
      return;
    }

    setIsProcessing(true);
    setError("");

    setOutputVideo("");
    setPersonFound(null);
    setMatchCount(0);
    setTimestamps([]);

    try {
      const formData = new FormData();

      formData.append("video", videoFile);
      formData.append("photo", photoFile);

      const response = await api.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Backend response:", response.data);

      setOutputVideo(response.data.video_url || "");
      setPersonFound(response.data.person_found ?? false);
      setMatchCount(response.data.match_count ?? 0);
      setTimestamps(response.data.timestamps ?? []);
    } catch (err) {
      console.error(err);
      setError("Something went wrong while processing the video.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="mx-auto max-w-5xl">

        {/* HEADER */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900">
            CCTV Person Tracker
          </h1>

          <p className="mt-2 text-gray-600">
            Upload CCTV footage and a reference photo to find the person.
          </p>
        </div>

        {/* UPLOAD SECTION */}
        <div className="grid gap-6 md:grid-cols-2">

          {/* VIDEO */}
          <div className="rounded-xl bg-white p-6 shadow">
            <div className="mb-4 flex items-center gap-3">
              <Video className="text-blue-600" size={28} />

              <div>
                <h2 className="text-xl font-semibold">
                  CCTV Video
                </h2>

                <p className="text-sm text-gray-500">
                  Upload the video to analyze
                </p>
              </div>
            </div>

            <label
              className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 p-8 transition ${
                isProcessing
                  ? "cursor-not-allowed opacity-50"
                  : "hover:border-blue-500"
              }`}
            >
              <Upload size={32} className="mb-3 text-gray-400" />

              <span className="text-sm text-gray-600">
                {videoFile
                  ? videoFile.name
                  : "Click to select CCTV video"}
              </span>

              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleVideoChange}
                disabled={isProcessing}
              />
            </label>
          </div>

          {/* PHOTO */}
          <div className="rounded-xl bg-white p-6 shadow">
            <div className="mb-4 flex items-center gap-3">
              <ImageIcon className="text-green-600" size={28} />

              <div>
                <h2 className="text-xl font-semibold">
                  Reference Photo
                </h2>

                <p className="text-sm text-gray-500">
                  Upload the person to search for
                </p>
              </div>
            </div>

            <label
              className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 p-8 transition ${
                isProcessing
                  ? "cursor-not-allowed opacity-50"
                  : "hover:border-green-500"
              }`}
            >
              <Upload size={32} className="mb-3 text-gray-400" />

              <span className="text-sm text-gray-600">
                {photoFile
                  ? photoFile.name
                  : "Click to select reference photo"}
              </span>

              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handlePhotoChange}
                disabled={isProcessing}
              />
            </label>
          </div>
        </div>

        {/* SUBMIT */}
        <div className="mt-6 text-center">
          <button
            onClick={handleSubmit}
            disabled={isProcessing}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-8 py-3 font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isProcessing ? (
              <>
                <Loader2 size={20} className="animate-spin" />
                Processing Video...
              </>
            ) : (
              <>
                <Upload size={20} />
                Submit for Analysis
              </>
            )}
          </button>
        </div>

        {/* ERROR */}
        {error && (
          <div className="mt-6 rounded-lg border border-red-300 bg-red-50 p-4 text-center text-red-700">
            {error}
          </div>
        )}

        {/* ANALYSIS RESULT */}
        {personFound !== null && !isProcessing && (
          <div className="mt-8 rounded-xl bg-white p-6 shadow">

            <h2 className="mb-6 text-center text-2xl font-bold text-gray-900">
              Analysis Result
            </h2>

            {/* PERSON FOUND */}
            <div
              className={`mb-6 rounded-lg p-5 text-center ${
                personFound
                  ? "bg-green-100 text-green-800"
                  : "bg-red-100 text-red-800"
              }`}
            >
              <p className="text-sm font-medium uppercase tracking-wide">
                Person Detection
              </p>

              <p className="mt-1 text-3xl font-bold">
                {personFound
                  ? "✓ PERSON FOUND"
                  : "✕ PERSON NOT FOUND"}
              </p>
            </div>

            {/* MATCH COUNT */}
            <div className="mb-6 rounded-lg bg-gray-50 p-5 text-center">
              <p className="text-sm font-medium text-gray-500">
                Match Count
              </p>

              <p className="mt-1 text-3xl font-bold text-gray-900">
                {matchCount}
              </p>
            </div>

            {/* TIMESTAMPS */}
            <div>
              <h3 className="mb-3 text-xl font-semibold text-gray-900">
                Detection Timestamps
              </h3>

              {timestamps.length > 0 ? (
                <div className="max-h-64 overflow-y-auto rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3">
                    {timestamps.map((timestamp, index) => (
                      <div
                        key={`${timestamp}-${index}`}
                        className="rounded-md bg-white px-3 py-2 text-center font-mono text-sm text-gray-700 shadow-sm"
                      >
                        {timestamp}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="rounded-lg bg-gray-50 p-4 text-center text-gray-500">
                  No matching timestamps found.
                </div>
              )}
            </div>
          </div>
        )}

        {/* PROCESSED VIDEO */}
        {outputVideo && (
          <div className="mt-8 rounded-xl bg-white p-6 shadow">

            <h2 className="mb-4 text-2xl font-bold text-gray-900">
              Processed Video
            </h2>

            <video
              controls
              className="w-full rounded-lg bg-black"
              src={outputVideo}
            />

            <div className="mt-4 text-center">
              <a
                href={outputVideo}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-lg bg-gray-800 px-5 py-2 font-medium text-white hover:bg-gray-900"
              >
                <ExternalLink size={18} />
                Open Video
              </a>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default Home;