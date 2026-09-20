def _set_remote_and_token():
  token = userdata.get('GITHUB_TOKEN')
  os.environ('GH_TOKEN') = token
  !git remote remove origin 2>/dev/null
  !git remote add origin https://github.com/{USERNAME}/{REPO_NAME}.git

def git_pull():
  _set_remote_and_token()
  !git -c credential.helper='!f() {{ echo "username=x-access-token"; echo "password=$GH_TOKEN"; }}; f' pull origin main
  print("✅ Pulled")

def git_push(commit_message):
  _set_remote_and_token()
  !git add . 
  !git commit -m "{commit_message}"
  !git -c credential.helper ='! f() {{ echo "username=x-access-token"; echo "password=$GH_TOKEN"; }}; f' push -q
  print("✅ Pushed")