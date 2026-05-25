import ShellLayout from '../components/ShellLayout';
import { Stethoscope, CheckCircle2, Microscope, CloudUpload, TriangleAlert, BarChart3, Upload } from 'lucide-react';
import { motion } from 'motion/react';

export default function Detection() {
  return (
    <ShellLayout>
      <header className="mb-12">
        <h2 className="text-3xl font-bold text-on-surface mb-2">Diagnostic Upload Center</h2>
        <p className="text-on-surface-variant font-medium">Select modality and upload medical imagery for deep learning analysis.</p>
      </header>

      <div className="grid grid-cols-12 gap-8">
        {/* Modality Selection Cards */}
        <div className="col-span-12 lg:col-span-4 space-y-8">
          <div className="group relative bg-white border-2 border-primary-container rounded-xl p-8 transition-all hover:shadow-lg cursor-pointer">
            <div className="flex justify-between items-start mb-6">
              <div className="h-14 w-14 bg-primary-container rounded-lg flex items-center justify-center text-white shadow-md">
                <Stethoscope className="w-8 h-8" />
              </div>
              <CheckCircle2 className="text-primary-container fill-primary-container/20 w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-on-surface mb-2">Detect Tuberculosis</h3>
            <p className="text-sm text-on-surface-variant/80 mb-6 leading-relaxed">Chest X-ray (CXR) interpretation using specialized neural networks.</p>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-surface-container-low rounded-full">
              <span className="w-2 h-2 bg-primary rounded-full"></span>
              <span className="text-[10px] font-bold text-primary tracking-widest uppercase">X-RAY ONLY</span>
            </div>
          </div>

          <div className="group relative bg-white border border-outline-variant/30 rounded-xl p-8 transition-all hover:border-outline-variant hover:shadow-md cursor-pointer">
            <div className="flex justify-between items-start mb-6">
              <div className="h-14 w-14 bg-surface-container-low rounded-lg flex items-center justify-center text-outline">
                <Microscope className="w-8 h-8" />
              </div>
            </div>
            <h3 className="text-xl font-bold text-on-surface mb-2">Detect Lung Cancer</h3>
            <p className="text-sm text-on-surface-variant/80 mb-6 leading-relaxed">Computed Tomography (CT) volumetric slice analysis.</p>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-surface-container-low rounded-full">
              <span className="w-2 h-2 bg-outline rounded-full"></span>
              <span className="text-[10px] font-bold text-on-surface-variant tracking-widest uppercase">CT SCAN ONLY</span>
            </div>
          </div>
        </div>

        {/* Upload and Preview Canvas */}
        <div className="col-span-12 lg:col-span-8">
          <div className="bg-white border border-outline-variant/30 rounded-xl overflow-hidden h-full flex flex-col shadow-sm">
            <div className="px-8 py-5 border-b border-outline-variant/30 flex items-center justify-between bg-surface-container-low/30">
              <span className="text-lg font-bold text-on-surface">Upload Medical Image</span>
              <span className="text-primary font-bold text-[10px] tracking-widest uppercase">MODALITY: X-RAY</span>
            </div>

            <div className="flex-1 p-10">
              <div className="h-full border-2 border-dashed border-outline-variant/50 rounded-2xl flex flex-col items-center justify-center p-12 bg-surface-container-low/10 transition-colors hover:bg-surface-container-low/30 cursor-pointer relative overflow-hidden group">
                <div className="absolute inset-0 flex items-center justify-center z-10 bg-surface-container-low/40 backdrop-blur-[2px]">
                  <div className="relative w-full h-full flex flex-col items-center justify-center gap-6">
                    <div className="bg-white border border-outline-variant/30 p-5 rounded-2xl shadow-xl flex flex-col items-center gap-2 max-w-xs text-center transform group-hover:scale-105 transition-transform">
                      <p className="text-[10px] font-bold text-outline opacity-60 tracking-wider">FILENAME</p>
                      <p className="font-medium text-sm text-on-surface font-mono">PATIENT_CXR_49102.DICOM</p>
                    </div>
                    <button className="bg-white text-primary border border-primary px-6 py-3 rounded-xl font-bold text-sm flex items-center gap-2 shadow-sm hover:bg-primary hover:text-white transition-all">
                      <Upload className="w-4 h-4" />
                      UPLOAD FROM DEVICE
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="px-10 py-8 bg-surface-container-low border-t border-outline-variant/30 flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-3 text-error">
                <TriangleAlert className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm font-bold">Image modality mismatch: Detected CT scan instead of X-ray.</span>
              </div>
              <div className="flex items-center gap-5 w-full md:w-auto">
                <button className="px-6 py-3 text-on-surface-variant font-bold text-sm hover:bg-white/50 rounded-lg transition-colors">Cancel</button>
                <button className="bg-primary-container text-white px-8 py-3.5 rounded-xl font-bold text-sm shadow-md hover:opacity-90 transition-all flex items-center gap-3 active:scale-95">
                  <span>Analyze Diagnostic Image</span>
                  <BarChart3 className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </ShellLayout>
  );
}
