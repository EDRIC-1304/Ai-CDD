import ShellLayout from '../components/ShellLayout';
import { Camera, Trash2, Edit2 } from 'lucide-react';
import { motion } from 'motion/react';

export default function Profile() {
  return (
    <ShellLayout>
      <div className="flex justify-between items-end mb-10">
        <div>
          <span className="text-xs font-bold text-primary uppercase mb-2 block tracking-widest">Clinician Directory</span>
          <h2 className="text-3xl font-bold text-on-surface">User Profile</h2>
          <p className="text-on-surface-variant mt-1 font-medium">Manage your account details and clinical credentials.</p>
        </div>
        <div className="flex gap-4">
          <button className="px-6 py-2.5 border border-outline-variant/50 text-on-surface font-semibold text-sm rounded-lg hover:bg-surface-container-low transition-colors outline-none">
            Discard
          </button>
          <button className="px-6 py-2.5 bg-primary-container text-white font-semibold text-sm rounded-lg hover:opacity-95 transition-opacity shadow-sm active:scale-[0.98]">
            Save Changes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-4">
          <div className="bg-white border border-outline-variant/30 rounded-2xl overflow-hidden shadow-sm">
            <div className="h-40 bg-primary-container/80 relative">
              <div className="absolute -bottom-16 left-8 h-32 w-32 rounded-3xl border-8 border-white overflow-hidden bg-white shadow-xl group">
                <img 
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuD02LJTRsIRGn-P05NX2ZaKxp8HIc-BpMcASWiSiGCJOhI2GEPoI4VcxVPr9hQh7OKhVi6_pAccWHZQ7qmCIOahktOWEMSpe_Co-o_n7kocC893Y2PoowBBlEf_hUZ3TPWNK7bm70XyCOMEIEdMqnyIM0s5l3aTJa5PSRIzKpHPXEfqJNUuABKv6-BNIDJjyTYi8Z06tMGeeMpk8QVPtFPLrWgiR_omXi21DLYS04Jk7hvPxUy5s3hKB0_ZVkUrg1-GqWXuvXM_4hLM" 
                  alt="Doctor Avatar" 
                  className="w-full h-full object-cover"
                  referrerPolicy="no-referrer"
                />
                <button className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 cursor-pointer">
                  <Camera className="text-white w-8 h-8" />
                </button>
              </div>
            </div>
            <div className="pt-20 pb-8 px-8">
              <div className="flex flex-col gap-1">
                <h3 className="text-2xl font-bold text-on-surface">Dr. Sarah Jenkins</h3>
                <p className="text-sm font-medium text-outline">s.jenkins@memorial-health.com</p>
              </div>
              <button className="mt-8 text-sm font-bold text-error flex items-center gap-2 hover:bg-error/5 py-2 px-3 rounded-lg transition-colors w-fit -ml-2">
                <Trash2 className="w-4 h-4" /> Remove Photo
              </button>
            </div>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-8">
          <div className="bg-white border border-outline-variant/30 rounded-2xl shadow-sm">
            <div className="px-8 py-6 border-b border-outline-variant/30 flex justify-between items-center">
              <h4 className="text-xl font-bold text-on-surface">Account Information</h4>
              <button className="text-primary font-bold text-sm flex items-center gap-2 hover:bg-primary/5 py-2 px-4 rounded-lg transition-colors">
                <Edit2 className="w-4 h-4" /> Edit details
              </button>
            </div>
            <div className="p-10 grid grid-cols-1 md:grid-cols-2 gap-10">
              <div className="space-y-3">
                <label className="text-[10px] font-bold text-outline uppercase tracking-wider block">Full Name</label>
                <div className="relative">
                  <input 
                    className="w-full bg-surface-container-low/30 border border-outline-variant/50 rounded-xl px-4 py-3.5 text-sm font-medium text-on-surface focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                    type="text" 
                    defaultValue="Dr. Sarah Jenkins"
                  />
                </div>
              </div>
              <div className="space-y-3">
                <label className="text-[10px] font-bold text-outline uppercase tracking-wider block">Email Address</label>
                <div className="relative">
                  <input 
                    className="w-full bg-surface-container-low/30 border border-outline-variant/50 rounded-xl px-4 py-3.5 text-sm font-medium text-on-surface focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                    type="email" 
                    defaultValue="s.jenkins@memorial-health.com"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </ShellLayout>
  );
}
