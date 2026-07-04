# Booting LLMOS under launchd

This makes the LLMOS kernel a real background service that starts at login and is
restarted if it dies — boot, init, and supervision delegated to macOS (per
IMPLEMENTATION.md). It is optional; you can also just run `python3 -m llmos.kerneld`
in a terminal.

## Install

```
# make the wrapper executable (once)
chmod +x ~/Code/LLMOS/bin/llmos-kernel

# link the agent into place and load it
cp ~/Code/LLMOS/launchd/com.mikeybeez.llmos.kernel.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mikeybeez.llmos.kernel.plist
```

The kernel now runs as `llmos-kernel` (legible in Login Items and `ps`), listening
on `~/Code/LLMOS/state/llmos-control.sock`.

## Use

```
python3 -m llmos.cli submit hello                 # watch it run live
python3 -m llmos.cli submit elevate --grant mem.write
python3 -m llmos.cli ps
python3 -m llmos.cli shutdown                     # ask the kernel to halt
```

## Uninstall

```
launchctl unload ~/Library/LaunchAgents/com.mikeybeez.llmos.kernel.plist
rm ~/Library/LaunchAgents/com.mikeybeez.llmos.kernel.plist
```

Logs are in `~/Code/LLMOS/state/logs/`.
