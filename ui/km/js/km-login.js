(function () {
  "use strict";
  const loginForm = document.getElementById("km-login-form");
  loginForm.elements.userId.value = "kmadmin";

  function showError(id, error) {
    const element = document.getElementById(id);
    element.textContent = error?.message || "操作失败";
    element.hidden = false;
  }

  const existing = KBotKmAuth.loadSession();
  if (existing?.access_token) location.replace("./dashboard.html");

  const previousFailure = KBotKmAuth.loadAuthFailure();
  if (previousFailure) {
    const request = previousFailure.request_id
      ? `；request_id: ${previousFailure.request_id}` : "";
    const path = previousFailure.path ? `；接口: ${previousFailure.path}` : "";
    showError(
      "login-error",
      new Error(`${previousFailure.message || previousFailure.code}${path}${request}`),
    );
  }

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    document.getElementById("login-error").hidden = true;
    const values = Object.fromEntries(new FormData(loginForm));
    try {
      await KBotKmAuth.login({
        user_id: values.userId.trim(),
        password: values.password,
      });
      KBotKmAuth.clearAuthFailure();
      location.replace("./dashboard.html");
    } catch (error) { showError("login-error", error); }
  });
})();
