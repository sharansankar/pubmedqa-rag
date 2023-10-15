import sys
from configs import DEMO_SESSION_CONFIG, PROD_SESSION_CONFIG
from session_builder import SessionBuilder

ARG_TO_CONFIG = {
  "demo": DEMO_SESSION_CONFIG,
  "prod": PROD_SESSION_CONFIG
}

if __name__ == "__main__":
  config_str = str(sys.argv[1])
  session_config = ARG_TO_CONFIG.get(config_str.lower(), DEMO_SESSION_CONFIG)

  session = SessionBuilder(session_config)

  while True:
    user_input = input(">> ")

    if user_input == "enable_debugging":
      print("**enabling debugging mode**")
      session.enable_debugging()
    elif user_input == "disable_debugging":
      print("**disabling debugging mode**")
      session.disable_debugging()
    elif user_input == "i_like_it":
      print("**noted**")
      session.response_is_upvoted()
    elif user_input == "exit_program":
      print("**exiting**")
      break
    else:
      response = session.process_user_prompt(user_input)
      print(f"LLM Agent: {response}")