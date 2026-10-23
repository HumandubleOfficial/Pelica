// Pelica's installer: one exe carrying the app (the payload.zip resource).
// It installs for the current user (no admin), by default into
// %LOCALAPPDATA%\Programs\Pelica, adds Start menu and desktop shortcuts that
// carry Pelica's AppUserModelID (so Windows shows its notifications) and an
// entry in Settings > Apps. An older version in the same folder is replaced.
// Built with /define:UNINSTALL (and no payload) it is the uninstaller, which
// removes exactly the files the installer listed; chats, settings and models
// stay unless the user asks (and then go to the Recycle Bin).
//
// Flags: /S (silent)  /D=<folder>  /NODESKTOP  /NOSHORTCUTS  /NOREG  /NOLAUNCH
//        /DATA (uninstall: also recycle the chats and settings)
//
// C# 5: it is compiled by the csc.exe that ships with Windows.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

static partial class Info
{
    public const string Name = "Pelica";
    public const string Aumid = "Pelica.Desktop";
    public const string Exe = "Pelica.exe";
    public const string Uninstaller = "Uninstall Pelica.exe";
    public const string ListFile = "uninstall.dat";
    public const string Key = @"Software\Microsoft\Windows\CurrentVersion\Uninstall\Pelica";
#if UNINSTALL
    public const bool IsUninstaller = true;
#else
    public const bool IsUninstaller = false;
#endif
}

class Opts
{
    public bool Silent, Desktop = true, Shortcuts = true, Reg = true, Launch = true, Data;
    public string Dir;

    public static Opts Parse(string[] args)
    {
        Opts o = new Opts();
        foreach (string a in args)
        {
            string u = a.ToUpperInvariant();
            if (u == "/S") o.Silent = true;
            else if (u == "/NODESKTOP") o.Desktop = false;
            else if (u == "/NOSHORTCUTS") o.Shortcuts = false;
            else if (u == "/NOREG") o.Reg = false;
            else if (u == "/NOLAUNCH") o.Launch = false;
            else if (u == "/DATA") o.Data = true;
            else if (u.StartsWith("/D=")) o.Dir = a.Substring(3).Trim('"');
        }
        if (o.Dir == null) o.Dir = Info.IsUninstaller ? Path.GetDirectoryName(Application.ExecutablePath) : Setup.DefaultDir();
        o.Dir = Path.GetFullPath(o.Dir);
        return o;
    }
}

static class Program
{
    [STAThread]
    static int Main(string[] args)
    {
        Opts o = Opts.Parse(args);
        if (o.Silent)
        {
            try
            {
                if (Info.IsUninstaller) Setup.Uninstall(o, null);
                else Setup.Install(o, null);
                Setup.RunCleanup();
                return 0;
            }
            catch (Exception e)
            {
                try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "pelica-setup.log"), DateTime.Now + " " + e + "\r\n"); } catch { }
                return 1;
            }
        }
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        // (for checking the design: each page drawn to a PNG, no window shown)
        foreach (string a in args)
            if (a.StartsWith("/SHOT=", StringComparison.OrdinalIgnoreCase))
            {
                new SetupForm(o).Shots(a.Substring(6).Trim('"'));
                return 0;
            }
        Application.Run(new SetupForm(o));
        Setup.RunCleanup();
        return 0;
    }
}

// ---------- the work ----------

static class Setup
{
    public static string DefaultDir()
    {
        // an existing install is updated where it is
        try
        {
            using (RegistryKey k = Registry.CurrentUser.OpenSubKey(Info.Key))
            {
                string loc = k == null ? null : k.GetValue("InstallLocation") as string;
                if (!string.IsNullOrEmpty(loc) && File.Exists(Path.Combine(loc, Info.Exe))) return loc;
            }
        }
        catch { }
        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", Info.Name);
    }

    /** Pelica processes started from this folder (the app; the engine is its child). */
    public static List<Process> Running(string dir)
    {
        List<Process> list = new List<Process>();
        string d = dir.TrimEnd('\\') + "\\";
        foreach (Process p in Process.GetProcessesByName("Pelica"))
        {
            try
            {
                if (p.MainModule.FileName.StartsWith(d, StringComparison.OrdinalIgnoreCase)) list.Add(p);
            }
            catch { }
        }
        return list;
    }

    static void CloseRunning(string dir)
    {
        foreach (Process p in Running(dir))
        {
            try
            {
                Process k = Process.Start(new ProcessStartInfo("taskkill", "/PID " + p.Id + " /T /F") { CreateNoWindow = true, UseShellExecute = false });
                k.WaitForExit(5000);
                p.WaitForExit(5000);
            }
            catch { }
        }
    }

    /** Retries a file operation a few times (a closing app can hold files for a moment). */
    static T Retry<T>(Func<T> f)
    {
        for (int i = 0; ; i++)
        {
            try { return f(); }
            catch (IOException) { if (i >= 20) throw; }
            catch (UnauthorizedAccessException) { if (i >= 20) throw; }
            Thread.Sleep(250);
        }
    }

    static string[] Listed(string dir)
    {
        string list = Path.Combine(dir, Info.ListFile);
        return File.Exists(list) ? File.ReadAllLines(list) : new string[0];
    }

    /** Deletes the files a list names (inside `dir` only), then the folders they leave empty. */
    static void RemoveListed(string dir, string[] files, string keep, Action<double, string> progress)
    {
        string root = dir.TrimEnd('\\') + "\\";
        HashSet<string> dirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < files.Length; i++)
        {
            if (files[i].Length == 0) continue;
            string f = Path.GetFullPath(Path.Combine(dir, files[i]));
            if (!f.StartsWith(root, StringComparison.OrdinalIgnoreCase)) continue;
            if (keep != null && string.Equals(f, keep, StringComparison.OrdinalIgnoreCase)) continue;
            for (string d = Path.GetDirectoryName(f); d.Length > root.Length; d = Path.GetDirectoryName(d)) dirs.Add(d);
            if (File.Exists(f))
            {
                string ff = f;
                Retry(() => { File.Delete(ff); return true; });
            }
            if (progress != null) progress((double)(i + 1) / files.Length, files[i]);
        }
        List<string> sorted = new List<string>(dirs);
        sorted.Sort(delegate(string a, string b) { return b.Length.CompareTo(a.Length); });
        foreach (string d in sorted)
        {
            try { if (Directory.Exists(d) && Directory.GetFileSystemEntries(d).Length == 0) Directory.Delete(d); } catch { }
        }
    }

    public static void Install(Opts o, Action<double, string> progress)
    {
        string dir = o.Dir;
        Directory.CreateDirectory(dir);
        CloseRunning(dir);
        DriveInfo drive = new DriveInfo(Path.GetPathRoot(dir));
        if (drive.AvailableFreeSpace < Info.Bytes + (64L << 20))
            throw new Exception(string.Format("There isn't enough space on {0}: Pelica needs about {1:0} MB.", drive.Name, Info.Bytes / 1048576.0));

        // the previous version's files go first (newer builds name some files differently)
        RemoveListed(dir, Listed(dir), null, null);

        List<string> written = new List<string>();
        string root = dir.TrimEnd('\\') + "\\";
        using (Stream s = Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip"))
        using (ZipArchive zip = new ZipArchive(s, ZipArchiveMode.Read))
        {
            long total = 0, done = 0;
            foreach (ZipArchiveEntry e in zip.Entries) total += e.Length;
            byte[] buf = new byte[1 << 20];
            foreach (ZipArchiveEntry e in zip.Entries)
            {
                string target = Path.GetFullPath(Path.Combine(dir, e.FullName.Replace('/', '\\')));
                if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase)) throw new Exception("bad path in the package: " + e.FullName);
                if (e.FullName.EndsWith("/")) continue;
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                string t = target;
                using (Stream src = e.Open())
                using (FileStream dst = Retry(() => new FileStream(t, FileMode.Create, FileAccess.Write)))
                {
                    int n;
                    while ((n = src.Read(buf, 0, buf.Length)) > 0)
                    {
                        dst.Write(buf, 0, n);
                        done += n;
                        if (progress != null) progress((double)done / Math.Max(1, total), e.FullName);
                    }
                }
                File.SetLastWriteTime(target, e.LastWriteTime.LocalDateTime);
                written.Add(e.FullName.Replace('/', '\\'));
            }
        }
        written.Add(Info.ListFile);
        File.WriteAllLines(Path.Combine(dir, Info.ListFile), written.ToArray());

        string exe = Path.Combine(dir, Info.Exe);
        if (o.Shortcuts)
        {
            Shortcuts.Create(Shortcuts.StartMenu(), exe, dir);
            if (o.Desktop) Shortcuts.Create(Shortcuts.Desktop(), exe, dir);
            else Shortcuts.RemoveIfOurs(Shortcuts.Desktop(), exe);
        }
        if (o.Reg) Register(dir, exe);
        if (progress != null) progress(1, "");
    }

    static void Register(string dir, string exe)
    {
        using (RegistryKey k = Registry.CurrentUser.CreateSubKey(Info.Key))
        {
            k.SetValue("DisplayName", Info.Name);
            k.SetValue("DisplayVersion", Info.Version);
            k.SetValue("Publisher", Info.Name);
            k.SetValue("DisplayIcon", exe + ",0");
            k.SetValue("InstallLocation", dir);
            k.SetValue("UninstallString", "\"" + Path.Combine(dir, Info.Uninstaller) + "\"");
            k.SetValue("QuietUninstallString", "\"" + Path.Combine(dir, Info.Uninstaller) + "\" /S");
            k.SetValue("EstimatedSize", (int)(Info.Bytes / 1024), RegistryValueKind.DWord);
            k.SetValue("InstallDate", DateTime.Now.ToString("yyyyMMdd"));
            k.SetValue("NoModify", 1, RegistryValueKind.DWord);
            k.SetValue("NoRepair", 1, RegistryValueKind.DWord);
        }
    }

    public static string UserData()
    {
        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), Info.Name);
    }

    public static void Uninstall(Opts o, Action<double, string> progress)
    {
        string dir = o.Dir;
        string exe = Path.Combine(dir, Info.Exe);
        string self = Application.ExecutablePath;
        string[] files = Listed(dir);
        // only a real Pelica folder: never sweep a folder this didn't install
        if (files.Length == 0 || !File.Exists(exe)) throw new Exception("Pelica isn't installed in " + dir + ".");
        CloseRunning(dir);

        Shortcuts.RemoveIfOurs(Shortcuts.StartMenu(), exe);
        Shortcuts.RemoveIfOurs(Shortcuts.Desktop(), exe);
        try
        {
            using (RegistryKey k = Registry.CurrentUser.OpenSubKey(Info.Key))
            {
                string loc = k == null ? null : k.GetValue("InstallLocation") as string;
                if (loc != null && string.Equals(Path.GetFullPath(loc).TrimEnd('\\'), dir.TrimEnd('\\'), StringComparison.OrdinalIgnoreCase))
                {
                    k.Close();
                    Registry.CurrentUser.DeleteSubKeyTree(Info.Key, false);
                }
            }
        }
        catch { }
        // "start with Windows" (Electron names the value after the AppUserModelID)
        foreach (string run in new string[] { @"Software\Microsoft\Windows\CurrentVersion\Run", @"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run" })
        {
            try
            {
                using (RegistryKey k = Registry.CurrentUser.OpenSubKey(run, true))
                {
                    if (k == null || k.GetValue(Info.Aumid) == null) continue;
                    string v = k.GetValue(Info.Aumid) as string;
                    if (v == null || v.IndexOf(dir, StringComparison.OrdinalIgnoreCase) >= 0) k.DeleteValue(Info.Aumid, false);
                }
            }
            catch { }
        }

        RemoveListed(dir, files, self, progress);

        if (o.Data && Directory.Exists(UserData()))
            Microsoft.VisualBasic.FileIO.FileSystem.DeleteDirectory(UserData(), Microsoft.VisualBasic.FileIO.UIOption.OnlyErrorDialogs, Microsoft.VisualBasic.FileIO.RecycleOption.SendToRecycleBin);

        // this exe can't delete itself while it runs: a moment after it exits, cmd
        // does (and removes the folder if nothing else is left in it)
        if (string.Equals(Path.GetDirectoryName(self), dir.TrimEnd('\\'), StringComparison.OrdinalIgnoreCase))
        {
            string cmd = "/c ping 127.0.0.1 -n 3 > nul & del /f /q \"" + self + "\" & rmdir \"" + dir.TrimEnd('\\') + "\"";
            Cleanup = cmd;
        }
    }

    /** What to run once this process has exited (the uninstaller removing itself). */
    public static string Cleanup;

    public static void RunCleanup()
    {
        if (Cleanup == null) return;
        try { Process.Start(new ProcessStartInfo("cmd.exe", Cleanup) { CreateNoWindow = true, UseShellExecute = false, WorkingDirectory = Path.GetTempPath() }); } catch { }
    }

    public static void Launch(string dir)
    {
        try { Process.Start(new ProcessStartInfo(Path.Combine(dir, Info.Exe)) { WorkingDirectory = dir, UseShellExecute = true }); } catch { }
    }
}

// ---------- shortcuts (with the AppUserModelID) ----------

static class Shortcuts
{
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class CShellLink { }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
    interface IShellLinkW
    {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder file, int cch, IntPtr fd, uint flags);
        void GetIDList(out IntPtr pidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder name, int cch);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder dir, int cch);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string dir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder args, int cch);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string args);
        void GetHotkey(out short key);
        void SetHotkey(short key);
        void GetShowCmd(out int cmd);
        void SetShowCmd(int cmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int cch, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string rel, uint reserved);
        void Resolve(IntPtr hwnd, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string file);
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint i, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct PropertyKey
    {
        public Guid Fmtid;
        public uint Pid;
    }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    struct PropVariant
    {
        [FieldOffset(0)] public ushort Vt;
        [FieldOffset(8)] public IntPtr Ptr;
    }

    public static string StartMenu()
    {
        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), Info.Name + ".lnk");
    }

    public static string Desktop()
    {
        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), Info.Name + ".lnk");
    }

    public static void Create(string link, string exe, string dir)
    {
        IShellLinkW sl = (IShellLinkW)new CShellLink();
        try
        {
            sl.SetPath(exe);
            sl.SetWorkingDirectory(dir);
            sl.SetIconLocation(exe, 0);
            sl.SetDescription("Local AI on your own GPU");
            // System.AppUserModel.ID: Windows ties the app's notifications to it
            IPropertyStore ps = (IPropertyStore)sl;
            PropertyKey key = new PropertyKey { Fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), Pid = 5 };
            PropVariant v = new PropVariant { Vt = 31, Ptr = Marshal.StringToCoTaskMemUni(Info.Aumid) };
            try
            {
                ps.SetValue(ref key, ref v);
                ps.Commit();
            }
            finally
            {
                Marshal.FreeCoTaskMem(v.Ptr);
            }
            Directory.CreateDirectory(Path.GetDirectoryName(link));
            ((IPersistFile)sl).Save(link, true);
        }
        finally
        {
            Marshal.ReleaseComObject(sl);
        }
    }

    /** Deletes a shortcut only if it starts this Pelica (not one the user pointed elsewhere). */
    public static void RemoveIfOurs(string link, string exe)
    {
        if (!File.Exists(link)) return;
        IShellLinkW sl = (IShellLinkW)new CShellLink();
        try
        {
            ((IPersistFile)sl).Load(link, 0);
            StringBuilder target = new StringBuilder(1024);
            sl.GetPath(target, target.Capacity, IntPtr.Zero, 0);
            if (string.Equals(target.ToString(), exe, StringComparison.OrdinalIgnoreCase)) File.Delete(link);
        }
        catch { }
        finally
        {
            Marshal.ReleaseComObject(sl);
        }
    }
}

// ---------- the window ----------

class SetupForm : Form
{
    enum Page { Options, Working, Done, Error }

    static readonly Color Bg = Color.FromArgb(31, 30, 29), Panel = Color.FromArgb(42, 41, 39), Line = Color.FromArgb(62, 62, 58),
        Text1 = Color.FromArgb(236, 235, 230), Text2 = Color.FromArgb(195, 193, 184), Muted = Color.FromArgb(155, 152, 144),
        Hot1 = Color.FromArgb(255, 178, 107), Hot2 = Color.FromArgb(229, 83, 61), Accent = Color.FromArgb(217, 119, 87);
    const int W = 620, H = 400;

    readonly Opts o;
    Page page = Page.Options;
    readonly float S;
    readonly Image logo;
    readonly Font fTitle, fSub, fHead, fBody, fSmall, fButton;
    volatile float prog;
    volatile string progFile = "";
    string error = "";
    bool optLaunch;
    string hover;
    readonly Dictionary<string, Rectangle> hits = new Dictionary<string, Rectangle>();
    readonly System.Windows.Forms.Timer tick = new System.Windows.Forms.Timer();

    [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);
    [DllImport("user32.dll")] static extern bool ReleaseCapture();
    [DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr hwnd, int msg, IntPtr w, IntPtr l);

    public SetupForm(Opts opts)
    {
        o = opts;
        optLaunch = o.Launch;
        S = DeviceDpi / 96f;
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.CenterScreen;
        ClientSize = new Size(D(W), D(H));
        BackColor = Bg;
        DoubleBuffered = true;
        KeyPreview = true;
        Text = Info.IsUninstaller ? "Uninstall Pelica" : "Pelica Setup";
        try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); } catch { }
        using (Stream s = Assembly.GetExecutingAssembly().GetManifestResourceStream("logo.png"))
            if (s != null) using (Image img = Image.FromStream(s)) logo = new Bitmap(img);
        fTitle = new Font("Segoe UI Semibold", 20f);
        fSub = new Font("Segoe UI", 10f);
        fHead = new Font("Segoe UI Semibold", 13f);
        fBody = new Font("Segoe UI", 10f);
        fSmall = new Font("Segoe UI", 8.75f);
        fButton = new Font("Segoe UI Semibold", 10f);
        tick.Interval = 33;
        tick.Tick += delegate { Invalidate(); };
    }

    int D(float v) { return (int)Math.Round(v * S); }

    public void Shots(string dir)
    {
        Directory.CreateDirectory(dir);
        IntPtr h = Handle;
        prog = 0.62f;
        progFile = @"resources\app\dist\assets\index.js";
        error = "There isn't enough space on C:\\: Pelica needs about 331 MB.";
        foreach (Page p in Enum.GetValues(typeof(Page)))
        {
            page = p;
            using (Bitmap b = new Bitmap(ClientSize.Width, ClientSize.Height))
            {
                DrawToBitmap(b, new Rectangle(Point.Empty, ClientSize));
                b.Save(Path.Combine(dir, (Info.IsUninstaller ? "uninstall-" : "setup-") + p + ".png"));
            }
        }
    }
    Rectangle R(float x, float y, float w, float h) { return new Rectangle(D(x), D(y), D(w), D(h)); }

    protected override CreateParams CreateParams
    {
        get
        {
            CreateParams cp = base.CreateParams;
            cp.ClassStyle |= 0x20000; // CS_DROPSHADOW
            return cp;
        }
    }

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        int round = 2; // Windows 11: rounded corners
        try { DwmSetWindowAttribute(Handle, 33, ref round, 4); } catch { }
    }

    // ---------- painting ----------

    static GraphicsPath Round(Rectangle r, float radius)
    {
        GraphicsPath p = new GraphicsPath();
        float d = radius * 2;
        p.AddArc(r.X, r.Y, d, d, 180, 90);
        p.AddArc(r.Right - d, r.Y, d, d, 270, 90);
        p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
        p.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
        p.CloseFigure();
        return p;
    }

    void Draw(Graphics g, string s, Font f, Color c, Rectangle r, bool wrap, StringAlignment align)
    {
        using (StringFormat sf = new StringFormat())
        using (SolidBrush b = new SolidBrush(c))
        {
            sf.Alignment = align;
            sf.LineAlignment = wrap ? StringAlignment.Near : StringAlignment.Center;
            sf.Trimming = wrap ? StringTrimming.Word : StringTrimming.EllipsisPath;
            if (!wrap) sf.FormatFlags = StringFormatFlags.NoWrap;
            g.DrawString(s, f, b, r, sf);
        }
    }

    void Primary(Graphics g, string key, string label, Rectangle r)
    {
        hits[key] = r;
        bool hot = hover == key;
        using (GraphicsPath p = Round(r, D(10)))
        using (LinearGradientBrush b = new LinearGradientBrush(r, hot ? Color.FromArgb(255, 192, 128) : Hot1, hot ? Color.FromArgb(240, 100, 76) : Hot2, LinearGradientMode.ForwardDiagonal))
            g.FillPath(b, p);
        Draw(g, label, fButton, Color.White, r, false, StringAlignment.Center);
    }

    void Ghost(Graphics g, string key, string label, Rectangle r)
    {
        hits[key] = r;
        using (GraphicsPath p = Round(r, D(10)))
        {
            using (SolidBrush b = new SolidBrush(hover == key ? Color.FromArgb(56, 55, 52) : Panel)) g.FillPath(b, p);
            using (Pen pen = new Pen(Line, 1)) g.DrawPath(pen, p);
        }
        Draw(g, label, fButton, Text1, r, false, StringAlignment.Center);
    }

    void Check(Graphics g, string key, string label, bool on, float x, float y)
    {
        Rectangle box = R(x, y, 18, 18);
        Size ts = TextRenderer.MeasureText(label, fBody);
        hits[key] = new Rectangle(box.X, box.Y - D(4), D(28) + ts.Width, D(26));
        using (GraphicsPath p = Round(box, D(5)))
        {
            if (on)
                using (LinearGradientBrush b = new LinearGradientBrush(box, Hot1, Hot2, LinearGradientMode.ForwardDiagonal)) g.FillPath(b, p);
            else
            {
                using (SolidBrush b = new SolidBrush(hover == key ? Color.FromArgb(56, 55, 52) : Panel)) g.FillPath(b, p);
                using (Pen pen = new Pen(Color.FromArgb(90, 89, 84), 1.2f * S)) g.DrawPath(pen, p);
            }
        }
        if (on)
            using (Pen pen = new Pen(Color.White, 2f * S) { StartCap = LineCap.Round, EndCap = LineCap.Round, LineJoin = LineJoin.Round })
                g.DrawLines(pen, new PointF[] { new PointF(box.X + 4.5f * S, box.Y + 9.5f * S), new PointF(box.X + 7.8f * S, box.Y + 12.8f * S), new PointF(box.X + 13.5f * S, box.Y + 5.8f * S) });
        Draw(g, label, fBody, hover == key ? Text1 : Text2, R(x + 28, y - 4, 520, 26), false, StringAlignment.Near);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        Graphics g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.InterpolationMode = InterpolationMode.HighQualityBicubic;
        g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
        hits.Clear();

        // a warm glow behind the mark
        using (GraphicsPath glow = new GraphicsPath())
        {
            glow.AddEllipse(R(-140, -170, 480, 380));
            using (PathGradientBrush b = new PathGradientBrush(glow))
            {
                b.CenterColor = Color.FromArgb(46, Hot2);
                b.SurroundColors = new Color[] { Color.FromArgb(0, Bg) };
                g.FillPath(b, glow);
            }
        }
        if (logo != null) g.DrawImage(logo, R(40, 40, 56, 56));
        Draw(g, Info.Name, fTitle, Text1, R(110, 36, 300, 40), false, StringAlignment.Near);
        Draw(g, Info.IsUninstaller ? "Uninstall" : "Local AI on your own GPU", fSub, Muted, R(112, 74, 400, 24), false, StringAlignment.Near);

        // close
        Rectangle close = new Rectangle(ClientSize.Width - D(46), 0, D(46), D(34));
        if (page != Page.Working)
        {
            hits["close"] = close;
            if (hover == "close") using (SolidBrush b = new SolidBrush(Color.FromArgb(196, 43, 28))) g.FillRectangle(b, close);
            using (Pen pen = new Pen(hover == "close" ? Color.White : Text2, 1.2f * S))
            {
                float cx = close.X + close.Width / 2f, cy = close.Y + close.Height / 2f, k = 5 * S;
                g.DrawLine(pen, cx - k, cy - k, cx + k, cy + k);
                g.DrawLine(pen, cx + k, cy - k, cx - k, cy + k);
            }
        }

        using (Pen sep = new Pen(Line, 1)) g.DrawLine(sep, D(40), D(H - 78), D(W - 40), D(H - 78));
        Rectangle primary = R(W - 40 - 140, H - 62, 140, 42);
        Rectangle secondary = R(W - 40 - 140 - 12 - 110, H - 62, 110, 42);

        switch (page)
        {
            case Page.Options:
                if (Info.IsUninstaller)
                {
                    Draw(g, "Remove Pelica from this PC?", fHead, Text1, R(40, 132, 540, 30), false, StringAlignment.Near);
                    Draw(g, "The app is removed from " + o.Dir + ". Your models are never touched.", fBody, Muted, R(40, 166, 540, 44), true, StringAlignment.Near);
                    Check(g, "data", "Also move my chats and settings to the Recycle Bin", o.Data, 40, 232);
                    Draw(g, "Version " + Info.Version, fSmall, Muted, R(40, H - 62, 260, 42), false, StringAlignment.Near);
                    Primary(g, "primary", "Uninstall", primary);
                    Ghost(g, "cancel", "Cancel", secondary);
                }
                else
                {
                    Draw(g, "Install to", fSmall, Muted, R(40, 128, 200, 20), false, StringAlignment.Near);
                    Rectangle box = R(40, 150, W - 80 - 112, 40);
                    using (GraphicsPath p = Round(box, D(9)))
                    {
                        using (SolidBrush b = new SolidBrush(Panel)) g.FillPath(b, p);
                        using (Pen pen = new Pen(Line, 1)) g.DrawPath(pen, p);
                    }
                    Draw(g, o.Dir, fBody, Text1, new Rectangle(box.X + D(12), box.Y, box.Width - D(24), box.Height), false, StringAlignment.Near);
                    Ghost(g, "change", "Change…", R(W - 40 - 100, 150, 100, 40));
                    Check(g, "desktop", "Create a desktop shortcut", o.Desktop, 40, 214);
                    Check(g, "launch", "Start Pelica when it's installed", optLaunch, 40, 246);
                    Draw(g, "Installs for you only, no admin needed. Your models, chats and settings stay where they are.", fSmall, Muted, R(40, 282, 540, 36), true, StringAlignment.Near);
                    Draw(g, string.Format("Version {0} · {1:0} MB", Info.Version, Info.Bytes / 1048576.0), fSmall, Muted, R(40, H - 62, 260, 42), false, StringAlignment.Near);
                    Primary(g, "primary", File.Exists(Path.Combine(o.Dir, Info.Exe)) ? "Update" : "Install", primary);
                }
                break;

            case Page.Working:
            {
                float p = Math.Max(0, Math.Min(1, prog));
                Draw(g, Info.IsUninstaller ? "Removing Pelica…" : "Installing Pelica…", fHead, Text1, R(40, 140, 420, 30), false, StringAlignment.Near);
                Draw(g, string.Format("{0:0}%", p * 100), fHead, Text2, R(W - 40 - 120, 140, 120, 30), false, StringAlignment.Far);
                Rectangle track = R(40, 184, W - 80, 6);
                using (GraphicsPath tp = Round(track, D(3)))
                using (SolidBrush b = new SolidBrush(Panel)) g.FillPath(b, tp);
                if (p > 0.005f)
                {
                    Rectangle fill = new Rectangle(track.X, track.Y, Math.Max(track.Height, (int)(track.Width * p)), track.Height);
                    using (GraphicsPath fp = Round(fill, D(3)))
                    using (LinearGradientBrush b = new LinearGradientBrush(track, Hot1, Hot2, LinearGradientMode.Horizontal)) g.FillPath(b, fp);
                }
                Draw(g, progFile, fSmall, Muted, R(40, 200, W - 80, 22), false, StringAlignment.Near);
                break;
            }

            case Page.Done:
            {
                Rectangle badge = R(40, 134, 40, 40);
                using (LinearGradientBrush b = new LinearGradientBrush(badge, Hot1, Hot2, LinearGradientMode.ForwardDiagonal)) g.FillEllipse(b, badge);
                using (Pen pen = new Pen(Color.White, 3f * S) { StartCap = LineCap.Round, EndCap = LineCap.Round, LineJoin = LineJoin.Round })
                    g.DrawLines(pen, new PointF[] { new PointF(badge.X + 11 * S, badge.Y + 21 * S), new PointF(badge.X + 17.5f * S, badge.Y + 27 * S), new PointF(badge.X + 29 * S, badge.Y + 14 * S) });
                if (Info.IsUninstaller)
                {
                    Draw(g, "Pelica was removed", fHead, Text1, R(96, 134, 480, 40), false, StringAlignment.Near);
                    Draw(g, o.Data ? "Your chats and settings are in the Recycle Bin. Your models were left alone." : "Your chats, settings and models were left alone, ready if you install it again.", fBody, Muted, R(96, 176, 480, 50), true, StringAlignment.Near);
                    Primary(g, "primary", "Close", primary);
                }
                else
                {
                    Draw(g, "Pelica is ready", fHead, Text1, R(96, 134, 480, 40), false, StringAlignment.Near);
                    Draw(g, "Find it in the Start menu. Once it's running, press Alt+Shift+Space anywhere to ask it something.", fBody, Muted, R(96, 176, 480, 50), true, StringAlignment.Near);
                    Check(g, "launch", "Start Pelica now", optLaunch, 96, 246);
                    Primary(g, "primary", "Finish", primary);
                }
                break;
            }

            case Page.Error:
                Draw(g, "Something went wrong", fHead, Text1, R(40, 132, 540, 30), false, StringAlignment.Near);
                Draw(g, error, fBody, Muted, R(40, 168, 540, 120), true, StringAlignment.Near);
                Primary(g, "primary", "Close", primary);
                break;
        }
    }

    // ---------- input ----------

    string HitAt(Point p)
    {
        foreach (KeyValuePair<string, Rectangle> kv in hits) if (kv.Value.Contains(p)) return kv.Key;
        return null;
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        string h = HitAt(e.Location);
        if (h != hover)
        {
            hover = h;
            Cursor = h != null ? Cursors.Hand : Cursors.Default;
            Invalidate();
        }
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);
        if (hover != null)
        {
            hover = null;
            Invalidate();
        }
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        // the empty top of the window moves it
        if (e.Button == MouseButtons.Left && HitAt(e.Location) == null && e.Y < D(120))
        {
            ReleaseCapture();
            SendMessage(Handle, 0xA1, (IntPtr)2, IntPtr.Zero);
        }
    }

    protected override void OnMouseClick(MouseEventArgs e)
    {
        base.OnMouseClick(e);
        if (e.Button == MouseButtons.Left) Act(HitAt(e.Location));
    }

    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (e.KeyCode == Keys.Enter) Act("primary");
        else if (e.KeyCode == Keys.Escape && page != Page.Working) Close();
    }

    void Act(string key)
    {
        if (key == null) return;
        switch (key)
        {
            case "close":
            case "cancel":
                Close();
                return;
            case "desktop":
                o.Desktop = !o.Desktop;
                break;
            case "launch":
                optLaunch = !optLaunch;
                break;
            case "data":
                o.Data = !o.Data;
                break;
            case "change":
                using (FolderBrowserDialog d = new FolderBrowserDialog())
                {
                    d.Description = "Choose where to install Pelica (a Pelica folder is made inside it).";
                    d.SelectedPath = Directory.Exists(o.Dir) ? o.Dir : Path.GetDirectoryName(o.Dir);
                    if (d.ShowDialog(this) == DialogResult.OK)
                    {
                        string p = d.SelectedPath;
                        // its own folder, so uninstalling never touches anything else
                        o.Dir = string.Equals(Path.GetFileName(p.TrimEnd('\\')), Info.Name, StringComparison.OrdinalIgnoreCase) ? p : Path.Combine(p, Info.Name);
                    }
                }
                break;
            case "primary":
                if (page == Page.Options) Start();
                else if (page == Page.Done || page == Page.Error)
                {
                    if (page == Page.Done && !Info.IsUninstaller && optLaunch) Setup.Launch(o.Dir);
                    Close();
                }
                return;
        }
        Invalidate();
    }

    void Start()
    {
        if (Setup.Running(o.Dir).Count > 0 &&
            MessageBox.Show(this, "Pelica is running. It will be closed so it can be " + (Info.IsUninstaller ? "removed." : "updated."), Text, MessageBoxButtons.OKCancel, MessageBoxIcon.Information) != DialogResult.OK)
            return;
        page = Page.Working;
        prog = 0;
        tick.Start();
        Invalidate();
        Thread t = new Thread(delegate()
        {
            string err = null;
            try
            {
                Action<double, string> report = delegate(double p, string f) { prog = (float)p; progFile = f; };
                if (Info.IsUninstaller) Setup.Uninstall(o, report);
                else Setup.Install(o, report);
            }
            catch (Exception ex)
            {
                err = ex.Message;
            }
            BeginInvoke((MethodInvoker)delegate
            {
                tick.Stop();
                if (err != null)
                {
                    error = err;
                    page = Page.Error;
                }
                else page = Page.Done;
                Invalidate();
            });
        });
        t.IsBackground = true;
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
    }
}
