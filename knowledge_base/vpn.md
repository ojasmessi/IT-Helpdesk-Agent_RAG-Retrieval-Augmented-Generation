# VPN (Remote Access) Troubleshooting

Contoso uses GlobalProtect VPN. The portal address is vpn.contoso.example.

## VPN will not connect
1. Check you have normal internet access (open any public website).
2. Make sure your laptop date and time are correct; wrong time breaks certificate checks.
3. Sign out of GlobalProtect, restart the laptop, and connect again.
4. If you see "Authentication failed", your password may have expired — see the password reset guide.
5. Hotel and café Wi-Fi often blocks VPN until you accept the Wi-Fi login page in a browser.

## VPN connects but internal sites do not load
Disconnect, then reconnect choosing the gateway "Europe-Dublin". Flush DNS with `ipconfig /flushdns` on Windows.

## Slow VPN
Video calls on Teams do not need the VPN. Close large downloads. Use a wired connection where possible.
