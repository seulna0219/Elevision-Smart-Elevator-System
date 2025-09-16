class LoginManager {
    constructor() {
        this.initializeEventListeners();
        this.addDemoAccountClickHandlers();
    }

    initializeEventListeners() {
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', this.handleLogin.bind(this));
        }
    }

    addDemoAccountClickHandlers() {
        const accountItems = document.querySelectorAll('.account-item');
        accountItems.forEach(item => {
            item.addEventListener('click', () => {
                const username = item.querySelector('.username').textContent;
                const password = item.querySelector('.password').textContent;
                
                document.getElementById('username').value = username;
                document.getElementById('password').value = password;
                
                // 添加視覺回饋
                item.style.background = '#d4edda';
                setTimeout(() => {
                    item.style.background = '#f8f9fa';
                }, 500);
            });
        });
    }

    async handleLogin(event) {
        event.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const submitBtn = document.querySelector('.login-btn');
        
        // 顯示載入狀態
        const originalText = submitBtn.textContent;
        submitBtn.textContent = '登入中...';
        submitBtn.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);
            
            const response = await fetch('/login', {
                method: 'POST',
                body: formData
            });
            
            if (response.redirected) {
                window.location.href = response.url;
            } else {
                window.location.reload();
            }
        } catch (error) {
            console.error('登入錯誤:', error);
            this.showMessage('系統錯誤，請稍後再試', 'error');
        } finally {
            // 恢復按鈕狀態
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        }
    }

    showMessage(message, type = 'error') {
        // 移除現有訊息
        const existingMessages = document.querySelector('.messages');
        if (existingMessages) {
            existingMessages.remove();
        }
        
        // 創建新訊息
        const messagesDiv = document.createElement('div');
        messagesDiv.className = 'messages';
        messagesDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
        
        // 插入到表單後面
        const loginForm = document.getElementById('loginForm');
        loginForm.insertAdjacentElement('afterend', messagesDiv);
        
        // 自動消失
        setTimeout(() => {
            messagesDiv.remove();
        }, 5000);
    }
}

// 初始化登入管理器
document.addEventListener('DOMContentLoaded', () => {
    new LoginManager();
});

// 視覺效果
document.addEventListener('DOMContentLoaded', () => {
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', () => {
            if (!input.value) {
                input.parentElement.classList.remove('focused');
            }
        });
    });
});
