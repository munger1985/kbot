(function () {
  "use strict";
  const loginSection = document.getElementById("login-section");
  const passwordSection = document.getElementById("password-section");
  const loginForm = document.getElementById("km-login-form");
  const passwordForm = document.getElementById("km-password-form");
  loginForm.elements.userId.value = "kmadmin";

  function showError(id, error) {
    const element = document.getElementById(id);
    element.textContent = error?.message || "操作失败";
    element.hidden = false;
  }

  const existing = KBotKmAuth.loadSession();
  if (existing?.access_token && !existing.must_change_password) location.replace("./dashboard.html");
  if (existing?.must_change_password) {
    loginSection.hidden = true;
    passwordSection.hidden = false;
  }

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    document.getElementById("login-error").hidden = true;
    const values = Object.fromEntries(new FormData(loginForm));
    try {
      const session = await KBotKmAuth.login({
        user_id: values.userId.trim(),
        password: values.password,
        domain_id: values.domainId ? Number(values.domainId) : null,
      });
      if (session.must_change_password) {
        loginSection.hidden = true;
        passwordSection.hidden = false;
        passwordForm.elements.currentPassword.value = values.password;
      } else {
        location.replace("./dashboard.html");
      }
    } catch (error) { showError("login-error", error); }
  });

  passwordForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    document.getElementById("password-error").hidden = true;
    const values = Object.fromEntries(new FormData(passwordForm));
    if (values.newPassword !== values.confirmPassword) {
      showError("password-error", new Error("两次输入的新密码不一致")); return;
    }
    try {
      await KBotKmAuth.changePassword({
        current_password: values.currentPassword,
        new_password: values.newPassword,
      });
      location.replace("./dashboard.html");
    } catch (error) { showError("password-error", error); }
  });
})();
