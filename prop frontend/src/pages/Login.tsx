import { Link } from 'react-router-dom';
import { Mail, Lock, Eye, Check } from 'lucide-react';
import { motion } from 'motion/react';

export default function Login() {
  return (
    <main className="min-h-screen flex">
      <section className="flex-1 flex flex-col justify-center px-8 md:px-16 lg:px-24 bg-surface">
        <div className="max-w-md w-full mx-auto">
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-primary-container rounded-lg flex items-center justify-center">
                <span className="material-symbols-outlined text-white">clinical_notes</span>
              </div>
              <h1 className="text-3xl font-bold text-on-surface">Chest Disease AI</h1>
            </div>
            <h2 className="text-xl font-semibold text-on-surface-variant">Welcome back</h2>
            <p className="text-on-surface-variant/70 mt-1">Please enter your credentials to access diagnostic tools.</p>
          </div>

          <form className="space-y-6" onSubmit={(e) => e.preventDefault()}>
            <div>
              <label className="text-xs font-bold text-on-surface uppercase mb-2 block tracking-wider" htmlFor="email">Email</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="text-outline w-4 h-4 group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full pl-10 pr-3 py-3 bg-white border border-outline-variant rounded-lg text-on-surface placeholder:text-outline focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                  id="email" 
                  type="email" 
                  placeholder="name@clinic.com"
                  required
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-xs font-bold text-on-surface uppercase block tracking-wider" htmlFor="password">Password</label>
                <Link to="#" className="text-xs font-bold text-primary hover:underline">Forgot password?</Link>
              </div>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="text-outline w-4 h-4 group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full pl-10 pr-10 py-3 bg-white border border-outline-variant rounded-lg text-on-surface placeholder:text-outline focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                  id="password" 
                  type="password" 
                  placeholder="••••••••"
                  required
                />
                <button type="button" className="absolute inset-y-0 right-0 pr-3 flex items-center">
                  <Eye className="text-outline w-4 h-4 hover:text-on-surface transition-colors" />
                </button>
              </div>
            </div>

            <div className="flex items-center">
              <input 
                id="remember"
                type="checkbox" 
                className="h-4 w-4 text-primary border-outline-variant rounded focus:ring-primary"
              />
              <label htmlFor="remember" className="ml-2 block text-sm text-on-surface-variant">
                Remember for 30 days
              </label>
            </div>

            <Link to="/dashboard">
              <button 
                type="submit"
                className="w-full bg-primary-container text-white py-4 px-4 rounded-xl font-semibold hover:bg-primary transition-colors shadow-sm active:scale-[0.98] mt-2"
              >
                Sign In to System
              </button>
            </Link>

            <p className="mt-6 text-center text-on-surface-variant">
              Don't have an account? <Link to="/register" className="text-primary font-bold hover:underline">Register now</Link>
            </p>
          </form>
        </div>
      </section>

      <section className="hidden lg:flex flex-1 relative bg-inverse-surface items-center justify-center p-12">
        <div className="absolute inset-0 z-0">
          <img 
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuDmCqMm5nvn0C9kvyI-3hEcHkA3044ZjpqbRWgb7s2xBk3v2-JgOwUoW1NaQiVuwIo_Qrsgc85BBWtWovW44HEEk2iTDAt9_OA4ai0-NHfE38L_Rr3FNo4DPHF-tCFc0MLd0mxO6JftKdvzvU95QQCeg3JYlL0oVeAY8IZvbucu2nadehfAX0fXfXNuNp6sJYgOhdl6ytn6PaZd3DzstgsNLxETnIEIH210m6qNeAhoE1pTx_apswQ5tCxAiMIn0TO_5hemekZkb5gw" 
            alt="Medical setting"
            className="w-full h-full object-cover opacity-40 mix-blend-luminosity"
            referrerPolicy="no-referrer"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-inverse-surface via-transparent to-transparent"></div>
        </div>
        
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
          className="relative z-10 max-w-lg w-full bg-surface-container-lowest/10 backdrop-blur-md border border-white/10 p-10 rounded-[40px]"
        >
          <div className="text-white">
            <h3 className="text-3xl font-bold leading-tight">An AI-Assisted Tuberculosis & Lung Cancer Screening System</h3>
          </div>
        </motion.div>
      </section>
    </main>
  );
}
