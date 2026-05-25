import { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, FileUp, History, User, LogOut, ClipboardList } from 'lucide-react';
import { motion } from 'motion/react';

interface SidebarProps {
  activeTitle?: string;
}

export default function Sidebar() {
  const navItems = [
    { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
    { icon: FileUp, label: 'Detection', path: '/detection' },
    { icon: History, label: 'Result History', path: '/history' },
    { icon: User, label: 'Profile', path: '/profile' },
  ];

  return (
    <aside className="fixed left-0 top-0 bottom-0 flex flex-col border-r border-outline-variant/30 bg-surface-container-lowest w-64 z-40">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-8 h-8 bg-primary-container rounded flex items-center justify-center">
            <ClipboardList className="text-white w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-black text-primary-container uppercase tracking-wider leading-none">Chest Disease AI</h2>
            <p className="text-[10px] text-outline mt-1 font-semibold uppercase tracking-widest">Clinical Diagnostic Tool</p>
          </div>
        </div>
      </div>
      
      <nav className="flex-1 px-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `
              flex items-center gap-3 px-4 py-3 transition-all duration-150 rounded-lg text-sm font-medium
              ${isActive 
                ? 'bg-primary-container/10 text-primary-container border-r-2 border-primary-container font-semibold' 
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-low'}
            `}
          >
            <item.icon className="w-5 h-5" />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="px-6 py-6 border-t border-outline-variant/30">
        <button className="flex items-center gap-3 px-4 py-2 text-error text-sm font-medium hover:bg-error/5 rounded-lg w-full transition-colors">
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </div>
    </aside>
  );
}
