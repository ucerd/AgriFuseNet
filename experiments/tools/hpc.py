"""Minimal SSH/SFTP helper for the Namal HPC cluster (password taken from the HPC_PASS environment variable).

python hpc.py run "<remote command>"
python hpc.py put <local> <remote>          (file or directory, recursive)
python hpc.py get <remote> <local>          (file or directory, recursive)
"""
import os
import sys
import stat
import posixpath
from pathlib import Path
import paramiko

HOST, USER = "10.0.0.154", "nabeel"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=os.environ["HPC_PASS"], timeout=20)
    return c


def run(c, cmd):
    _, out, err = c.exec_command(cmd, timeout=600)
    o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
    code = out.channel.recv_exit_status()
    sys.stdout.write(o)
    if e.strip():
        sys.stdout.write("[stderr] " + e)
    return code


def put(sftp, local, remote):
    local = Path(local)
    if local.is_dir():
        try:
            sftp.mkdir(remote)
        except IOError:
            pass
        for p in local.iterdir():
            put(sftp, p, posixpath.join(remote, p.name))
    else:
        sftp.put(str(local), remote)


def get(sftp, remote, local):
    if stat.S_ISDIR(sftp.stat(remote).st_mode):
        Path(local).mkdir(parents=True, exist_ok=True)
        for e in sftp.listdir_attr(remote):
            get(sftp, posixpath.join(remote, e.filename), str(Path(local) / e.filename))
    else:
        sftp.get(remote, str(local))


if __name__ == "__main__":
    c = connect()
    if sys.argv[1] == "run":
        sys.exit(run(c, sys.argv[2]))
    sftp = c.open_sftp()
    (put if sys.argv[1] == "put" else get)(sftp, sys.argv[2], sys.argv[3])
    print("ok")
