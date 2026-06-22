# UCL Research Computing: Cluster Login Guide

This guide explains how to connect to UCL's high-performance computing (HPC) clusters (e.g., Myriad, Kathleen, Aristotle).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Logging in (Linux / macOS)](#logging-in-linux--macos)
3. [Logging in (Windows)](#logging-in-windows)
4. [Logging in from Outside UCL](#logging-in-from-outside-ucl)
5. [Transferring Data](#transferring-data)
6. [Job Scheduling (qsub, qstat)](#job-scheduling)
7. [Resource Monitoring](#resource-monitoring)
8. [Graphical Applications (X-Forwarding)](#graphical-applications-x-forwarding)

---

## Prerequisites
Logging in is most straightforward if you are inside the UCL firewall. If you are outside (e.g., at home), you MUST:
- Use the **UCL VPN**, OR
- Connect via the **UCL SSH Gateway** (`ssh-gateway.ucl.ac.uk`).

---

## Quick-Start Commands
| Cluster | Address | Commands |
| :--- | :--- | :--- |
| **Myriad** | `myriad.rc.ucl.ac.uk` | `ssh <ucl_id>@myriad.rc.ucl.ac.uk` |
| **Kathleen** | `kathleen.rc.ucl.ac.uk` | `ssh kathleen` |
| **Aristotle** | `aristotle.rc.ucl.ac.uk` | `ssh aristotle` |

---

## Pro-User Setup (Completed)
We have configured your Mac for the fastest possible access.

### 1. SSH Config (Shortcuts)
Your shortcuts are stored in `~/.ssh/config`. You can now use these simple aliases instead of full addresses:
- `ssh myriad`
- `ssh kathleen`
- `ssh gateway`

### 2. Passwordless Login (SSH Keys)
You have generated an ED25519 security key and shared it with the clusters.
- **Mac Command:** `ssh-keygen -t ed25519`
- **Link Command:** `ssh-copy-id myriad` (and `kathleen`)
- **Result:** You no longer need to enter your UCL password to log in from this Mac.

### 3. Apple Keychain (True Automation)
We have added your SSH Key to the **Apple Keychain** so that the AI (and yourself) can log in without entering a passphrase:
- **Command:** `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`
- **Audit Result:** Hostnames `login12.myriad.ucl.ac.uk` and `login01` (Kathleen) were successfully fetched automatically!

### 4. Locale Warning Fix
To prevent Perl/Locale errors on login, we have added `SendEnv -LC_* -LANG` to your SSH configuration.

---

**Example:**
```bash
ssh ccxxxxx@myriad.rc.ucl.ac.uk
```

> [!TIP]
> If you get "broken pipe" or timeout errors on macOS, try:
> `ssh -o ConnectTimeout=10 <ucl_id>@<system_name>.rc.ucl.ac.uk`

---

## Logging in (Windows)
- **OpenSSH (Cmd/PowerShell):** Works just like Linux/macOS.
- **PuTTY:** 
  1. Host Name: `<system_name>.rc.ucl.ac.uk`
  2. Connection type: SSH
  3. When prompted, enter your username (without the `.rc.ucl.ac.uk` part).
- **WSL2:** Use the Linux instructions. Avoid using VPN inside WSL as it may break connectivity; use the SSH Gateway instead.

---

## Logging in from Outside UCL
External logins require SSH Keys for the Gateway.

1. **Via Gateway:**
   ```bash
   ssh <ucl_id>@ssh-gateway.ucl.ac.uk
   # Then from the gateway:
   ssh <system_name>
   ```

## Cluster "Personalities" (Lessons Learned)
When submitting jobs, choose your cluster based on your computational needs:

### **1. Myriad (The Versatile Engine)**
- **Workload:** General-purpose, Python, AI/ML (GPUs), and single-core tasks.
- **Strength:** Nodes with **1.5 TB RAM** (Type B) and **NVIDIA GPUs**.
- **Rule:** It will accept small jobs (1 core, 4GB RAM).

### **2. Kathleen (The Multi-Node Monster)**
- **Workload:** High-Throughput Parallel jobs (MPI).
- **Strength:** Massive scaling across multiple nodes.
- **Rule:** **Jobs do NOT share nodes.** If you request 1 core, it might reject you.
- **Policy:** For best results, request **at least 2 nodes (80 cores/MPI 80)** to satisfy the parallel-only policy.

---

## Job Submission Cheat Sheet
### **Myriad (General)**
```bash
qsub -l h_rt=1:00:00 -l mem=4G submit_job.sh
```

### **Kathleen (Parallel)**
```bash
qsub -l h_rt=1:00:00 -l mem=2G -pe mpi 80 submit_job.sh
```

---

**Local to Cluster:**
```bash
scp <local_file> <ucl_id>@<system_name>.rc.ucl.ac.uk:~/Scratch/
```

**Cluster to Local:**
```bash
scp <ucl_id>@<system_name>.rc.ucl.ac.uk:~/Scratch/<file> <local_path>
```

---

## Job Scheduling
Most clusters use the Sun Grid Engine (SGE) scheduler.

- **Submit a job:** `qsub myscript.sh`
- **Check job status:** `qstat`
- **Delete a job:** `qdel <job_id>`
- **Explain errors:** `qexplain <job_id>` (useful for `Eqw` states)

---

## Resource Monitoring
- **Quota:** `lquota`
- **Disk Usage:** `du -ch <dir>` or `du -h --max-depth=1`
- **Active Node Usage:** `nodesforjob <job_id>`
- **Finished Job History:** `jobhist`

---

## Graphical Applications (X-Forwarding)
To run GUIs (like MATLAB, nedit, etc.), you need an X-Server:
- **macOS:** Install [XQuartz](https://www.xquartz.org/).
- **Windows:** Use Xming or Exceed.

Connect with the `-X` flag:
```bash
ssh -X <ucl_id>@<system_name>.rc.ucl.ac.uk
```
Test by running `nedit`.

---

## How to Log Out
Type `exit`, `logout`, or press `Ctrl+D`.
