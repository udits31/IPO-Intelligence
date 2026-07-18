import './globals.css';

export const metadata = {
  title: 'OpenBell | IPO Intelligence for Indian Investors',
  description: 'Live IPO research, transparent decision signals, comparisons, and post-listing portfolio monitoring.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
