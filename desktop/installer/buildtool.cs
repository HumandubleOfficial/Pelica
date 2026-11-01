// Build helpers for the installer (C# 5, compiled with the csc.exe that ships
// with Windows, so building needs no downloads):
//   buildtool stamp <exe> <ico> <version> <product> <description> <company> <copyright> <original name>
//     gives an exe (Pelica.exe, a renamed electron.exe) Pelica's icon and
//     version information, so Explorer, the taskbar and Task Manager show Pelica
//   buildtool zip <folder> <out.zip>
//     packs a folder (the installed app, as the installer will lay it out)
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;

static class BuildTool
{
    static int Main(string[] a)
    {
        try
        {
            if (a.Length == 9 && a[0] == "stamp") Stamp(a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
            else if (a.Length == 3 && a[0] == "zip") Zip(a[1], a[2]);
            else
            {
                Console.Error.WriteLine("usage: buildtool stamp <exe> <ico> <version> <product> <description> <company> <copyright> <original name> | zip <folder> <out.zip>");
                return 2;
            }
            return 0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("buildtool: " + e.Message);
            return 1;
        }
    }

    // ---------- zip ----------

    static void Zip(string dir, string output)
    {
        string root = Path.GetFullPath(dir).TrimEnd('\\');
        string[] files = Directory.GetFiles(root, "*", SearchOption.AllDirectories);
        Array.Sort(files, StringComparer.OrdinalIgnoreCase);
        long bytes = 0;
        using (FileStream fs = new FileStream(output, FileMode.Create, FileAccess.Write))
        using (ZipArchive zip = new ZipArchive(fs, ZipArchiveMode.Create))
        {
            foreach (string f in files)
            {
                string rel = f.Substring(root.Length + 1).Replace('\\', '/');
                ZipArchiveEntry e = zip.CreateEntry(rel, CompressionLevel.Optimal);
                e.LastWriteTime = File.GetLastWriteTime(f);
                using (Stream dst = e.Open())
                using (FileStream src = File.OpenRead(f))
                {
                    src.CopyTo(dst);
                    bytes += src.Length;
                }
            }
        }
        Console.WriteLine(string.Format("zip: {0} files, {1:0.0} MB -> {2:0.0} MB", files.Length, bytes / 1048576.0, new FileInfo(output).Length / 1048576.0));
    }

    // ---------- resources ----------

    const uint LOAD_AS_DATA = 0x2 | 0x20;
    static readonly IntPtr RT_ICON = (IntPtr)3, RT_GROUP_ICON = (IntPtr)14, RT_VERSION = (IntPtr)16;

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr LoadLibraryEx(string file, IntPtr h, uint flags);
    [DllImport("kernel32")] static extern bool FreeLibrary(IntPtr h);
    delegate bool EnumNameProc(IntPtr h, IntPtr type, IntPtr name, IntPtr param);
    delegate bool EnumLangProc(IntPtr h, IntPtr type, IntPtr name, ushort lang, IntPtr param);
    [DllImport("kernel32", CharSet = CharSet.Unicode)] static extern bool EnumResourceNames(IntPtr h, IntPtr type, EnumNameProc f, IntPtr p);
    [DllImport("kernel32", CharSet = CharSet.Unicode)] static extern bool EnumResourceLanguages(IntPtr h, IntPtr type, IntPtr name, EnumLangProc f, IntPtr p);
    [DllImport("kernel32")] static extern IntPtr FindResourceEx(IntPtr h, IntPtr type, IntPtr name, ushort lang);
    [DllImport("kernel32")] static extern IntPtr LoadResource(IntPtr h, IntPtr r);
    [DllImport("kernel32")] static extern IntPtr LockResource(IntPtr r);
    [DllImport("kernel32")] static extern uint SizeofResource(IntPtr h, IntPtr r);
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr BeginUpdateResource(string file, bool deleteExisting);
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool UpdateResource(IntPtr h, IntPtr type, IntPtr name, ushort lang, byte[] data, uint size);
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool UpdateResource(IntPtr h, IntPtr type, string name, ushort lang, byte[] data, uint size);
    [DllImport("kernel32", SetLastError = true)] static extern bool EndUpdateResource(IntPtr h, bool discard);

    /** A resource's name: a number or a string. */
    class Res
    {
        public int Id = -1;
        public string Str;
        public ushort Lang;
        public byte[] Data;
    }

    static bool IsInt(IntPtr p) { return ((ulong)p.ToInt64() >> 16) == 0; }

    static List<Res> List(IntPtr mod, IntPtr type, bool withData)
    {
        List<Res> names = new List<Res>();
        EnumResourceNames(mod, type, delegate(IntPtr h, IntPtr t, IntPtr name, IntPtr p)
        {
            Res r = new Res();
            if (IsInt(name)) r.Id = (int)name.ToInt64();
            else r.Str = Marshal.PtrToStringUni(name);
            names.Add(r);
            return true;
        }, IntPtr.Zero);
        List<Res> all = new List<Res>();
        foreach (Res n in names)
        {
            IntPtr namePtr = n.Str == null ? (IntPtr)n.Id : Marshal.StringToHGlobalUni(n.Str);
            try
            {
                EnumResourceLanguages(mod, type, namePtr, delegate(IntPtr h, IntPtr t, IntPtr nm, ushort lang, IntPtr p)
                {
                    Res r = new Res();
                    r.Id = n.Id;
                    r.Str = n.Str;
                    r.Lang = lang;
                    if (withData)
                    {
                        IntPtr found = FindResourceEx(mod, type, namePtr, lang);
                        uint size = SizeofResource(mod, found);
                        r.Data = new byte[size];
                        Marshal.Copy(LockResource(LoadResource(mod, found)), r.Data, 0, (int)size);
                    }
                    all.Add(r);
                    return true;
                }, IntPtr.Zero);
            }
            finally
            {
                if (n.Str != null) Marshal.FreeHGlobal(namePtr);
            }
        }
        return all;
    }

    static void Put(IntPtr upd, IntPtr type, Res r, byte[] data)
    {
        bool ok = r.Str == null
            ? UpdateResource(upd, type, (IntPtr)r.Id, r.Lang, data, data == null ? 0u : (uint)data.Length)
            : UpdateResource(upd, type, r.Str, r.Lang, data, data == null ? 0u : (uint)data.Length);
        if (!ok) throw new Exception("UpdateResource failed: " + Marshal.GetLastWin32Error());
    }

    static void Stamp(string exe, string icoPath, string version, string product, string description, string company, string copyright, string original)
    {
        IntPtr mod = LoadLibraryEx(exe, IntPtr.Zero, LOAD_AS_DATA);
        if (mod == IntPtr.Zero) throw new Exception("can't open " + exe + " (" + Marshal.GetLastWin32Error() + ")");
        List<Res> groups, versions;
        try
        {
            groups = List(mod, RT_GROUP_ICON, true);
            versions = List(mod, RT_VERSION, false);
        }
        finally
        {
            FreeLibrary(mod);
        }

        // the new icon: its images, and a group directory pointing at them
        byte[] ico = File.ReadAllBytes(icoPath);
        int count = BitConverter.ToUInt16(ico, 4);
        MemoryStream grp = new MemoryStream();
        BinaryWriter gw = new BinaryWriter(grp);
        gw.Write((ushort)0);
        gw.Write((ushort)1);
        gw.Write((ushort)count);
        List<byte[]> images = new List<byte[]>();
        for (int i = 0; i < count; i++)
        {
            int e = 6 + i * 16;
            int size = BitConverter.ToInt32(ico, e + 8);
            int offset = BitConverter.ToInt32(ico, e + 12);
            byte[] img = new byte[size];
            Array.Copy(ico, offset, img, 0, size);
            images.Add(img);
            gw.Write(ico, e, 12); // width, height, colors, reserved, planes, bit count, bytes
            gw.Write((ushort)(i + 1));
        }

        ushort lang = groups.Count > 0 ? groups[0].Lang : (ushort)1033;
        IntPtr upd = BeginUpdateResource(exe, false);
        if (upd == IntPtr.Zero) throw new Exception("can't write " + exe + " (" + Marshal.GetLastWin32Error() + ")");
        bool done = false;
        try
        {
            // out with the old icons (each group and the images it lists)
            foreach (Res g in groups)
            {
                int n = BitConverter.ToUInt16(g.Data, 4);
                for (int i = 0; i < n; i++)
                {
                    Res img = new Res();
                    img.Id = BitConverter.ToUInt16(g.Data, 6 + i * 14 + 12);
                    img.Lang = g.Lang;
                    Put(upd, RT_ICON, img, null);
                }
                Put(upd, RT_GROUP_ICON, g, null);
            }
            for (int i = 0; i < images.Count; i++)
            {
                Res img = new Res();
                img.Id = i + 1;
                img.Lang = lang;
                Put(upd, RT_ICON, img, images[i]);
            }
            // (under the first group's name: Windows shows the first group)
            Res ng = new Res();
            ng.Id = groups.Count > 0 ? groups[0].Id : 1;
            ng.Str = groups.Count > 0 ? groups[0].Str : null;
            ng.Lang = lang;
            Put(upd, RT_GROUP_ICON, ng, grp.ToArray());

            foreach (Res v in versions) Put(upd, RT_VERSION, v, null);
            Res nv = new Res();
            nv.Id = 1;
            nv.Lang = 1033;
            Put(upd, RT_VERSION, nv, VersionInfo(version, product, description, company, copyright, original));
            done = true;
        }
        finally
        {
            if (!EndUpdateResource(upd, !done) && done) throw new Exception("EndUpdateResource failed: " + Marshal.GetLastWin32Error());
        }
        Console.WriteLine("stamp: " + Path.GetFileName(exe) + " -> " + product + " " + version + ", " + count + " icon sizes");
    }

    // ---------- VS_VERSIONINFO ----------

    class Blocks
    {
        public MemoryStream Ms = new MemoryStream();
        public BinaryWriter W;
        public Blocks() { W = new BinaryWriter(Ms); }
        public void Pad() { while (Ms.Position % 4 != 0) W.Write((byte)0); }
        public long Begin(string key, int valueLength, int type)
        {
            Pad();
            long start = Ms.Position;
            W.Write((ushort)0);
            W.Write((ushort)valueLength);
            W.Write((ushort)type);
            W.Write(Encoding.Unicode.GetBytes(key + "\0"));
            Pad();
            return start;
        }
        public void End(long start)
        {
            long end = Ms.Position;
            Ms.Position = start;
            W.Write((ushort)(end - start));
            Ms.Position = end;
        }
    }

    static byte[] VersionInfo(string version, string product, string description, string company, string copyright, string original)
    {
        string[] p = (version + ".0.0.0").Split('.');
        uint v0 = uint.Parse(p[0]), v1 = uint.Parse(p[1]), v2 = uint.Parse(p[2]), v3 = uint.Parse(p[3]);
        uint ms = (v0 << 16) | v1, ls = (v2 << 16) | v3;
        Blocks b = new Blocks();
        long root = b.Begin("VS_VERSION_INFO", 52, 0);
        b.W.Write(0xFEEF04BDu); // signature
        b.W.Write(0x00010000u); // structure version
        b.W.Write(ms);
        b.W.Write(ls);
        b.W.Write(ms);
        b.W.Write(ls);
        b.W.Write(0x3Fu); // flags mask
        b.W.Write(0u); // flags
        b.W.Write(0x40004u); // VOS_NT_WINDOWS32
        b.W.Write(1u); // VFT_APP
        b.W.Write(0u);
        b.W.Write(0u);
        b.W.Write(0u);
        long sfi = b.Begin("StringFileInfo", 0, 1);
        long table = b.Begin("040904b0", 0, 1);
        string[,] kv = {
            { "CompanyName", company }, { "FileDescription", description }, { "FileVersion", version },
            { "InternalName", product }, { "LegalCopyright", copyright }, { "OriginalFilename", original },
            { "ProductName", product }, { "ProductVersion", version },
        };
        for (int i = 0; i < kv.GetLength(0); i++)
        {
            long s = b.Begin(kv[i, 0], kv[i, 1].Length + 1, 1);
            b.W.Write(Encoding.Unicode.GetBytes(kv[i, 1] + "\0"));
            b.End(s);
        }
        b.End(table);
        b.End(sfi);
        long vfi = b.Begin("VarFileInfo", 0, 1);
        long tr = b.Begin("Translation", 4, 0);
        b.W.Write((ushort)0x0409);
        b.W.Write((ushort)0x04B0);
        b.End(tr);
        b.End(vfi);
        b.End(root);
        return b.Ms.ToArray();
    }
}
