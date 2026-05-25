import { Link } from 'react-router-dom';
import { User, Mail, Lock, Verified, ArrowRight, ClipboardList } from 'lucide-react';
import { motion } from 'motion/react';

export default function Register() {
  return (
    <main className="min-h-screen flex flex-col md:flex-row">
      <aside className="hidden md:flex md:w-5/12 lg:w-1/2 relative bg-primary-container items-center justify-center p-8 overflow-hidden">
        <div className="absolute inset-0 z-0">
          <img 
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuAWDqx-cN7IJd09A_KJNQZ1xqpHmlGEpWPUhuHCuwKri2X8nHLeNve1xDEA6XQwGTAfk8m58_Inj3Ewd8Iju1xoAw0gN2tF8maMO8BEIzdGGYXl8N7uT6rll4I6gEsBVCaj_Nn8-HH-mD1PrOja3cDNoMc45ui8ZAssDzORfsZzBxwvtZk6RCztg1qYVlJ_rhMyHmgq5xKpdEUZwdefzkjVHxdxwy7FH6VfW7Yh_NO2NdoF_DyZrurg9zBx6d1AD0bIY67f-RSHmrHl" 
            alt="Clinical diagnostic environment"
            className="w-full h-full object-cover opacity-30 mix-blend-overlay"
            referrerPolicy="no-referrer"
          />
        </div>
        <div className="relative z-10 max-w-lg text-white">
          <div className="mb-10">
            <span className="inline-flex items-center justify-center p-3 bg-white/10 rounded-xl backdrop-blur-md mb-6">
              <ClipboardList className="w-10 h-10" />
            </span>
            <h1 className="text-5xl font-bold mb-6 leading-tight">Chest Disease Detection System</h1>
          </div>
        </div>
        <div className="absolute bottom-0 left-0 w-full h-1/3 bg-gradient-to-t from-primary/40 to-transparent pointer-events-none"></div>
      </aside>

      <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-md">
          <header className="mb-10">
            <div className="flex items-center gap-2 mb-6 md:hidden">
              <ClipboardList className="text-primary w-6 h-6" />
              <span className="text-xs font-bold tracking-widest text-primary uppercase">Chest Disease AI</span>
            </div>
            <h2 className="text-3xl font-semibold text-on-surface mb-2">Create Account</h2>
            <p className="text-sm text-on-surface-variant">Register to start performing clinical diagnostic scans.</p>
          </header>

          <form className="space-y-6" onSubmit={(e) => e.preventDefault()}>
            <div>
              <label className="block text-xs font-bold text-on-surface mb-2 uppercase tracking-wider" htmlFor="name">Full Name</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="text-outline w-5 h-5 group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full pl-10 pr-3 py-3 bg-surface-container-lowest border border-outline-variant rounded-lg text-on-surface placeholder:text-outline focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                  id="name" 
                  placeholder="Dr. Julian Vane" 
                  required 
                  type="text"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-on-surface mb-2 uppercase tracking-wider" htmlFor="email">Email</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="text-outline w-5 h-5 group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full pl-10 pr-3 py-3 bg-surface-container-lowest border border-outline-variant rounded-lg text-on-surface placeholder:text-outline focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                  id="email" 
                  placeholder="name@clinic.org" 
                  required 
                  type="email"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-on-surface mb-2 uppercase tracking-wider" htmlFor="password">Password</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="text-outline w-5 h-5 group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full pl-10 pr-3 py-3 bg-surface-container-lowest border border-outline-variant rounded-lg text-on-surface placeholder:text-outline focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                  id="password" 
                  placeholder="••••••••" 
                  required 
                  type="password"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-on-surface mb-2 uppercase tracking-wider" htmlFor="confirm-password">Confirm Password</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Verified className="text-outline w-5 h-5 group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full pl-10 pr-3 py-3 bg-surface-container-lowest border border-outline-variant rounded-lg text-on-surface placeholder:text-outline focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                  id="confirm-password" 
                  placeholder="••••••••" 
                  required 
                  type="password"
                />
              </div>
            </div>

            <div className="flex items-start gap-3 py-2">
              <input id="terms" type="checkbox" required className="mt-1 h-4 w-4 rounded border-outline-variant text-primary focus:ring-primary" />
              <label htmlFor="terms" className="text-sm text-on-surface-variant">
                I agree to the <Link to="#" className="text-primary font-medium hover:underline">Terms of Service</Link> and <Link to="#" className="text-primary font-medium hover:underline">Privacy Policy</Link> concerning medical data handling.
              </label>
            </div>

            <div className="pt-2">
              <Link to="/dashboard">
                <button className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-primary text-white font-semibold rounded-lg hover:bg-primary/90 active:scale-[0.98] transition-all shadow-sm">
                  Create Account <ArrowRight className="w-5 h-5" />
                </button>
              </Link>
            </div>
          </form>

          <footer className="mt-10 pt-10 border-t border-outline-variant/30 text-center">
            <p className="text-sm text-on-surface-variant">
              Already registered for clinical use? 
              <Link to="/login" className="text-primary font-bold hover:underline ml-1">Sign In</Link>
            </p>
          </footer>
        </div>
      </main>
    </main>
  );
}
