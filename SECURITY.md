# Security & Safety

## Safety disclaimer

**Pukeko Robot Controller is an educational project.** It lets a large language
model autonomously drive a *physical* robot based on what it sees through a
webcam. The model can and will make mistakes.

Run it only:

- in a **controlled environment** — a clear, bounded surface (e.g. a table with
  margins or barriers) the robot cannot fall off or escape from, away from
  people, pets, fragile objects, and other hazards;
- under **direct adult supervision**, with someone ready to physically stop or
  power off the robot at all times;
- with realistic expectations — the agent is experimental and is **not**
  intended for production, unattended, or safety-critical use.

The "Emergency stop" button halts the *agent*, but it cannot interrupt a motion
the robot's firmware is already executing. **Physical supervision — and the
power switch — are the real safety mechanism, not the software.**

This software is provided "as is", without warranty of any kind (see
[LICENSE](./LICENSE)). You are responsible for operating the hardware safely.

## Reporting a vulnerability

If you discover a security vulnerability, please report it privately rather than
opening a public issue:

- Use GitHub's **"Report a vulnerability"** (private security advisory) under the
  repository's **Security** tab.

We'll acknowledge the report and work on a fix. Please allow a reasonable window
to address the issue before any public disclosure.
