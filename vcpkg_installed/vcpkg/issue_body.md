Package: bzip2[core,tool]:x64-linux@1.0.8#5

**Host Environment**

- Host: x64-linux
- Compiler: GNU 13.3.0
-    vcpkg-tool version: 2024-07-10-d2dfc73769081bdd9b782d08d27794780b7a99b9
    vcpkg-scripts version: 1de2026f28 2024-07-12 (1 year, 6 months ago)

**To Reproduce**

`vcpkg install `

**Failure logs**

```
-- Using cached bzip2-1.0.8.tar.gz.
-- Cleaning sources at /home/kh/my_paper/champsim/src/champsim/vcpkg/buildtrees/bzip2/src/bzip2-1-336d4794a3.clean. Use --editable to skip cleaning for the packages you specify.
-- Extracting source /home/kh/my_paper/champsim/src/champsim/vcpkg/downloads/bzip2-1.0.8.tar.gz
-- Applying patch fix-import-export-macros.patch
-- Using source at /home/kh/my_paper/champsim/src/champsim/vcpkg/buildtrees/bzip2/src/bzip2-1-336d4794a3.clean
-- Found external ninja('1.11.1').
-- Configuring x64-linux
-- Building x64-linux-dbg
-- Building x64-linux-rel
-- Fixing pkgconfig file: /home/kh/my_paper/champsim/src/champsim/vcpkg/packages/bzip2_x64-linux/lib/pkgconfig/bzip2.pc
-- Fixing pkgconfig file: /home/kh/my_paper/champsim/src/champsim/vcpkg/packages/bzip2_x64-linux/debug/lib/pkgconfig/bzip2.pc
-- Installing: /home/kh/my_paper/champsim/src/champsim/vcpkg/packages/bzip2_x64-linux/share/bzip2/copyright
-- Downloading https://github.com/NixOS/patchelf/releases/download/0.14.5/patchelf-0.14.5-x86_64.tar.gz -> patchelf-0.14.5-x86_64.tar.gz...
[DEBUG] To include the environment variables in debug output, pass --debug-env
[DEBUG] Trying to load bundleconfig from /home/kh/my_paper/champsim/src/champsim/vcpkg/vcpkg-bundle.json
[DEBUG] Failed to open: /home/kh/my_paper/champsim/src/champsim/vcpkg/vcpkg-bundle.json
[DEBUG] Bundle config: readonly=false, usegitregistry=false, embeddedsha=nullopt, deployment=Git, vsversion=nullopt
[DEBUG] Metrics enabled.
[DEBUG] Feature flag 'binarycaching' unset
[DEBUG] Feature flag 'compilertracking' unset
[DEBUG] Feature flag 'registries' unset
[DEBUG] Feature flag 'versions' unset
[DEBUG] Feature flag 'dependencygraph' unset
[DEBUG] 1000: execute_process(curl --fail -L https://github.com/NixOS/patchelf/releases/download/0.14.5/patchelf-0.14.5-x86_64.tar.gz --create-dirs --output /home/kh/my_paper/champsim/src/champsim/vcpkg/downloads/patchelf-0.14.5-x86_64.tar.gz.490867.part)
[DEBUG] 1000: cmd_execute_and_stream_data() returned 6 after     4168 us
error: Missing patchelf-0.14.5-x86_64.tar.gz and downloads are blocked by x-block-origin.
error: https://github.com/NixOS/patchelf/releases/download/0.14.5/patchelf-0.14.5-x86_64.tar.gz: curl failed to download with exit code 6
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed

  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0curl: (6) Could not resolve host: github.com

[DEBUG] /source/src/vcpkg/base/downloads.cpp(1030): 
[DEBUG] Time in subprocesses: 4168us
[DEBUG] Time in parsing JSON: 3us
[DEBUG] Time in JSON reader: 0us
[DEBUG] Time in filesystem: 128us
[DEBUG] Time in loading ports: 0us
[DEBUG] Exiting after 4.74 ms (4446us)

CMake Error at scripts/cmake/vcpkg_download_distfile.cmake:32 (message):
      
      Failed to download file with error: 1
      If you are using a proxy, please check your proxy setting. Possible causes are:
      
      1. You are actually using an HTTP proxy, but setting HTTPS_PROXY variable
         to `https://address:port`. This is not correct, because `https://` prefix
         claims the proxy is an HTTPS proxy, while your proxy (v2ray, shadowsocksr
         , etc..) is an HTTP proxy. Try setting `http://address:port` to both
         HTTP_PROXY and HTTPS_PROXY instead.
      
      2. If you are using Windows, vcpkg will automatically use your Windows IE Proxy Settings
         set by your proxy software. See https://github.com/microsoft/vcpkg-tool/pull/77
         The value set by your proxy might be wrong, or have same `https://` prefix issue.
      
      3. Your proxy's remote server is out of service.
      
      If you've tried directly download the link, and believe this is not a temporary
      download server failure, please submit an issue at https://github.com/Microsoft/vcpkg/issues
      to report this upstream download server failure.
      

Call Stack (most recent call first):
  scripts/cmake/vcpkg_download_distfile.cmake:270 (z_vcpkg_download_distfile_show_proxy_and_fail)
  scripts/cmake/vcpkg_find_acquire_program.cmake:170 (vcpkg_download_distfile)
  scripts/cmake/z_vcpkg_fixup_rpath.cmake:96 (vcpkg_find_acquire_program)
  scripts/ports.cmake:196 (z_vcpkg_fixup_rpath_in_dir)



```

**Additional context**

<details><summary>vcpkg.json</summary>

```
{
  "name": "champsim",
  "version-string": "1.0",
  "dependencies": [
    "cli11",
    "nlohmann-json",
    "fmt",
    "bzip2",
    "liblzma",
    "zlib",
    "catch2"
  ]
}

```
</details>
