import pyautogui
import time


def mouse_click():
  while True:
    time.sleep(30)
    pyautogui.click()


js_method = """
function ConnectButton(){
  // console.log("keep connecting"); 
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
}
intervalId = setInterval(ConnectButton, 60000);
// 18212
clearInterval(intervalId);
"""


if __name__ == '__main__':
  mouse_click()
