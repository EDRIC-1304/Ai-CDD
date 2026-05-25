import ShellLayout from '../components/ShellLayout';
import { SearchIcon, ChevronLeft, ChevronRight, MoreHorizontal } from 'lucide-react';

export default function ResultHistory() {
  const records = [
    { 
      id: '#SCAN-9402', 
      patient: 'Johnathan Miller', 
      date: 'Oct 24, 2023', 
      time: '14:20 PM', 
      type: 'Pneumonia', 
      result: 'Positive', 
      isPositive: true,
      img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAqT0JIfeWKLi24Qz1Fbh1bbItmHogGW5-figRVqT2NZGkVBrlUZ48lGo9qWOg7V4c0dunWchsLqYZzwuSWswmpRyRNOEflwH6drvjs3_z01reE54GoqMoQyWOXwixDQUITW68coae0eTPTQSnZaArGezXsCDzKEBorNcKgbd96D6Ca_DZDp0hlUa9eYoZEX-7oAk0tR6YYGoBOr3fMa_RoBY7cAYwLRaNLOzQb2nyFYEs_92262td6kKpilDhu73HcTKVRWNDbcTBJ'
    },
    { 
      id: '#SCAN-9398', 
      patient: 'Sarah Wilson', 
      date: 'Oct 24, 2023', 
      time: '11:05 AM', 
      type: 'Normal', 
      result: 'Negative', 
      isPositive: false,
      img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCt8PEf860V0o5l4GC1JBUWaDq5-41qqB5Fjs4Hlgel2JayLgKXY4NgFae1_7sN8SXhDUBDzf_675a7DTR1dG6vgunwK6LBLGlh4js02gqEYNxSd3xM5W9QCYvUA6GjgRSlj5NYUHd6aPWZ8MuCUy0e7bVbPfPyVDw8ccmJILDRA6iy2yXdJs3YYLGJLN1ter_uI6NcL6Z6TJglwkSAQQZMZGI_n4pkaqfeclZBrKe_wcF1Ujc4SuyYpcfgVVnK695AKKNzj4-8hxRX'
    },
    { 
      id: '#SCAN-9382', 
      patient: 'Marcus Davies', 
      date: 'Oct 23, 2023', 
      time: '09:15 AM', 
      type: 'Effusion', 
      result: 'Positive', 
      isPositive: true,
      img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBtpA4j1sDx2rNlb15hhzXqRgTQB4ufWVgorPbfJSqOJ-8N53N4uFb7LX47S5ZaY-TLSlLG3U_Rl_QZnKynbwaHiZpZA5OWsLSp-mHeS6_rDRxYeSTiXPMftT6f3fP3JZg9TByjfQunDmq4LhHdRz0Gwv9XChmSMWu-b-ASiTED0Jt0ZoRqwZ-wfR_jDi446tw_PCxxatjCsiAJ3w-zm0QZCPmYkHrzA-l28o3y9Wy_7EJOeRNPD4h8Ik_0x58-fCYU2DnSiv44TPsG'
    }
  ];

  return (
    <ShellLayout>
      <div className="flex items-end justify-between mb-10">
        <div>
          <h1 className="text-3xl font-bold text-on-surface">Result History</h1>
          <p className="text-on-surface-variant mt-1">Manage and review clinical detection records.</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-10">
        <div className="bg-white border border-outline-variant/30 p-6 rounded-xl shadow-sm">
          <div className="text-outline font-bold text-xs mb-1 tracking-widest">TOTAL SCANS</div>
          <div className="text-3xl font-bold text-on-surface">1,284</div>
        </div>
        <div className="bg-white border border-outline-variant/30 p-6 rounded-xl shadow-sm">
          <div className="text-outline font-bold text-xs mb-1 tracking-widest">POSITIVE CASES</div>
          <div className="text-3xl font-bold text-error">142</div>
        </div>
      </div>

      <div className="bg-white border border-outline-variant/30 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-container-low border-b border-outline-variant/30">
                <th className="px-6 py-4 text-xs font-bold text-outline tracking-widest">SCAN ID / PATIENT</th>
                <th className="px-6 py-4 text-xs font-bold text-outline tracking-widest">DATE & TIME</th>
                <th className="px-6 py-4 text-xs font-bold text-outline tracking-widest">DISEASE TYPE</th>
                <th className="px-6 py-4 text-xs font-bold text-outline tracking-widest">RESULT</th>
                <th className="px-6 py-4 text-xs font-bold text-outline tracking-widest text-right">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/20">
              {records.map((record) => (
                <tr key={record.id} className="group hover:bg-surface-container-low/50 transition-colors cursor-pointer">
                  <td className="px-6 py-5">
                    <div className="flex items-center gap-4">
                      <div className="h-14 w-14 bg-black border border-outline-variant/30 rounded-lg overflow-hidden shrink-0 shadow-sm">
                        <img 
                          src={record.img} 
                          alt="Scan thumbnail" 
                          className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                        />
                      </div>
                      <div>
                        <div className="text-sm font-bold text-on-surface">{record.id}</div>
                        <div className="text-xs text-outline font-medium">Patient: {record.patient}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-5">
                    <div className="text-sm font-semibold text-on-surface">{record.date}</div>
                    <div className="text-xs text-outline">{record.time}</div>
                  </td>
                  <td className="px-6 py-5">
                    <span className="text-sm font-bold text-on-surface">{record.type}</span>
                  </td>
                  <td className="px-6 py-5">
                    <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                      record.isPositive 
                        ? 'bg-error-container/40 text-error' 
                        : 'bg-tertiary/10 text-tertiary'
                    }`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${record.isPositive ? 'bg-error' : 'bg-tertiary'}`}></span>
                      {record.result}
                    </div>
                  </td>
                  <td className="px-6 py-5 text-right">
                    <button className="text-primary font-bold text-sm hover:underline">View Details</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="px-6 py-5 bg-white border-t border-outline-variant/30 flex items-center justify-between">
          <button className="px-4 py-2 border border-outline-variant/30 rounded-lg text-sm font-bold text-on-surface/70 hover:bg-surface-container-low transition-all">
            Previous
          </button>
          <div className="flex gap-2">
            <button className="w-10 h-10 rounded-lg bg-primary-container/10 text-primary-container font-bold text-sm shadow-sm ring-1 ring-primary-container/30">1</button>
            <button className="w-10 h-10 rounded-lg text-outline font-bold text-sm hover:bg-surface-container-low transition-all">2</button>
            <button className="w-10 h-10 rounded-lg text-outline font-bold text-sm hover:bg-surface-container-low transition-all">3</button>
            <div className="w-10 h-10 flex items-center justify-center text-outline">
              <MoreHorizontal className="w-4 h-4" />
            </div>
            <button className="w-10 h-10 rounded-lg text-outline font-bold text-sm hover:bg-surface-container-low transition-all">12</button>
          </div>
          <button className="px-4 py-2 border border-outline-variant/30 rounded-lg text-sm font-bold text-on-surface/70 hover:bg-surface-container-low transition-all">
            Next
          </button>
        </div>
      </div>
    </ShellLayout>
  );
}
