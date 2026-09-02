param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Command)
$ErrorActionPreference = 'Stop'
if (-not $Command -or $Command.Count -eq 0) { throw 'Usage: remote_run.ps1 COMMAND [ARG...]' }
$root = (git rev-parse --show-toplevel).Trim()
if (git -C $root status --porcelain --untracked-files=all) { throw 'Local tree is dirty' }
$sha = (git -C $root rev-parse HEAD).Trim()
$null = git -C $root fetch origin
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
$upstream = (git -C $root rev-parse '@{upstream}').Trim()
if ($sha -ne $upstream) { throw 'HEAD is not pushed' }
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + $sha.Substring(0,12) + '-' + ([guid]::NewGuid().ToString('N').Substring(0,8))
$json = ConvertTo-Json -Compress -InputObject @($Command)
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
& ssh -p 22003 qingchan@asimov1.cs.uga.edu /home/qingchan/work/concept-flow/scripts/server_run.sh $runId $sha $payload
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
