import ShellLayout from '../components/ShellLayout';
import { List, CloudUpload, TriangleAlert, ArrowRight } from 'lucide-react';
import { motion } from 'motion/react';

export default function Dashboard() {
  return (
    <ShellLayout>
      <div className="flex items-end justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-on-surface">Practitioner Dashboard</h1>
          <p className="text-on-surface-variant mt-1">Review diagnostic metrics and recent patient scans.</p>
        </div>
        <div className="flex gap-4">
          <button className="px-6 py-2.5 border border-outline-variant rounded-lg font-semibold text-primary bg-white hover:bg-surface-container-low transition-all clinical-shadow flex items-center gap-2">
            <List className="w-5 h-5" />
            View History
          </button>
          <button className="px-6 py-2.5 bg-primary-container text-white rounded-lg font-semibold hover:opacity-90 transition-all clinical-shadow flex items-center gap-2">
            <CloudUpload className="w-5 h-5" />
            Quick Upload
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Distribution Chart Card */}
        <div className="col-span-12 lg:col-span-7 bg-white border border-outline-variant/30 rounded-xl p-8 clinical-shadow">
          <div className="flex justify-between items-center mb-8">
            <h3 className="text-xl font-semibold text-on-surface">Total Scans Distribution</h3>
            <div className="flex items-center gap-4 text-xs font-bold text-on-surface-variant">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-primary-container"></span>TB Detected</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-tertiary"></span>Cancer Detected</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-outline-variant"></span>Normal</span>
            </div>
          </div>
          
          <div className="flex items-center justify-around">
            <div className="relative w-64 h-64">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                <circle cx="18" cy="18" fill="transparent" r="15.915" stroke="#EAECF0" strokeWidth="4"></circle>
                <circle cx="18" cy="18" fill="transparent" r="15.915" stroke="#026AA2" strokeWidth="4" strokeDasharray="45 55" strokeDashoffset="0"></circle>
                <circle cx="18" cy="18" fill="transparent" r="15.915" stroke="#005930" strokeWidth="4" strokeDasharray="25 75" strokeDashoffset="-45"></circle>
                <circle cx="18" cy="18" fill="transparent" r="15.915" stroke="#c0c7d1" strokeWidth="4" strokeDasharray="30 70" strokeDashoffset="-70"></circle>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-4xl font-bold">1,248</span>
                <span className="text-[10px] font-bold text-on-surface-variant uppercase tracking-tighter mt-1">Total Scans</span>
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="flex flex-col">
                <span className="text-3xl font-bold text-primary-container leading-none">562</span>
                <span className="text-xs text-on-surface-variant mt-1 uppercase font-bold tracking-wider">Tuberculosis Cases</span>
              </div>
              <div className="flex flex-col">
                <span className="text-3xl font-bold text-tertiary leading-none">312</span>
                <span className="text-xs text-on-surface-variant mt-1 uppercase font-bold tracking-wider">Lung Cancer Detected</span>
              </div>
              <div className="flex flex-col">
                <span className="text-3xl font-bold text-outline leading-none">374</span>
                <span className="text-xs text-on-surface-variant mt-1 uppercase font-bold tracking-wider">Normal Results</span>
              </div>
            </div>
          </div>
        </div>

        {/* Latest Analysis Card */}
        <div className="col-span-12 lg:col-span-5">
          <div className="bg-white border border-outline-variant/30 rounded-xl overflow-hidden clinical-shadow h-full flex flex-col">
            <div className="px-6 py-4 border-b border-outline-variant/30 bg-surface-container-low flex justify-between items-center">
              <h3 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Latest Analysis</h3>
            </div>
            <div className="p-8 flex-1">
              <div className="flex gap-6 mb-8">
                <div className="w-28 h-28 bg-black rounded-lg border border-outline-variant/30 overflow-hidden flex-shrink-0 shadow-lg">
                  <img 
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuBo87i1wKA-NTDcG6BgbXvZ9SkZJ8jgxuz_V0KddYD2Iqg3G3fM1HzQirSqIQ-VSEmRKWRP4hNTjt7booV7f-zegCSRIp9XcBZmcKTrQLg6Vw8CXc16KRzIYFtcq73ouTTbksGnfmNoFU7VJKBUa1IyD0d3lhoo8A096hNvuRpY7u8WS32iMrQT2L7ZIkhm5jSuQu16XCZR79I15LeUYNMjcmEYpbfBigu1CHMcUb6s22dLXUYgb7tKo6XCDyY4hsVQhZIVp4Rk_ETG" 
                    alt="Chest X-Ray" 
                    className="w-full h-full object-cover opacity-80"
                    referrerPolicy="no-referrer"
                  />
                </div>
                <div className="flex-1">
                  <div className="text-lg font-bold text-on-surface">Patient: ID-98421-B</div>
                  <div className="text-sm text-on-surface-variant mt-1">Uploaded 12 mins ago</div>
                  <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-error-container/50 text-error text-xs font-bold">
                    <TriangleAlert className="w-3.5 h-3.5" />
                    TB Detection Likely (89%)
                  </div>
                </div>
              </div>
              <button className="w-full flex items-center justify-between group py-4 border-t border-outline-variant/30 mt-auto">
                <span className="text-primary font-bold text-sm">Review Full Report</span>
                <ArrowRight className="text-primary w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </ShellLayout>
  );
}
