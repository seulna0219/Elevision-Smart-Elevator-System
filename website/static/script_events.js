class EventsApp {
    constructor() {
        this.events = [];
        this.filteredEvents = [];
        this.initializeElements();
        this.attachEventListeners();
        this.updateNavigation();
        this.loadEvents();
    }

    initializeElements() {
        // Header elements
        this.refreshBtn = document.getElementById('refreshBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.eventCount = document.getElementById('eventCount');
        
        // Summary elements
        this.totalEvents = document.getElementById('totalEvents');
        this.latestTime = document.getElementById('latestTime');
        this.objectTypes = document.getElementById('objectTypes');
        
        // Filter elements
        this.dataSource = document.getElementById('dataSource');
        this.objectFilter = document.getElementById('objectFilter');
        
        // Table elements
        this.loadingEvents = document.getElementById('loadingEvents');
        this.noEvents = document.getElementById('noEvents');
        this.eventsTable = document.getElementById('eventsTable');
        this.eventsTableBody = document.getElementById('eventsTableBody');
        
        // Other elements
        this.toastContainer = document.getElementById('toastContainer');
    }

    attachEventListeners() {
        if (this.refreshBtn) {
            this.refreshBtn.addEventListener('click', () => this.loadEvents());
        }
        
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.clearEvents());
        }
        
        if (this.dataSource) {
            this.dataSource.addEventListener('change', () => this.loadEvents());
        }
        
        if (this.objectFilter) {
            this.objectFilter.addEventListener('change', () => this.applyFilters());
        }
    }

    updateNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const currentPath = window.location.pathname;
        
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === currentPath) {
                item.classList.add('active');
            }
        });
    }

    async loadEvents() {
        try {
            this.showLoading(true);
            
            // 取得選擇的資料來源
            const source = this.dataSource ? this.dataSource.value : 'memory';
            const response = await fetch(`/events_data?source=${source}&limit=100`);
            const data = await response.json();
            
            this.events = data.events || [];
            this.filteredEvents = [...this.events];
            
            // 顯示資料來源資訊
            const sourceText = data.source === 'database' ? 'MongoDB 資料庫' : '即時記憶體';
            console.log(`已從 ${sourceText} 載入 ${this.events.length} 個事件`);
            
            this.updateSummary();
            this.updateFilters();
            this.displayEvents();
            
            if (this.events.length === 0) {
                this.showNoEvents();
            } else {
                this.showEventsTable();
            }
            
        } catch (error) {
            console.error('Load events error:', error);
            this.showToast('error', '載入失敗', '無法載入事件紀錄');
            this.showNoEvents();
        } finally {
            this.showLoading(false);
        }
    }

    async clearEvents() {
        // 顯示確認對話框
        if (!confirm('確定要清除所有事件記錄嗎？此操作無法復原。')) {
            return;
        }

        try {
            this.showLoading(true);
            
            const response = await fetch('/clear_events');
            const data = await response.json();
            
            if (data.success) {
                this.events = [];
                this.filteredEvents = [];
                this.updateSummary();
                this.displayEvents();
                this.showNoEvents();
                this.showToast('success', '清除成功', data.message);
            } else {
                this.showToast('error', '清除失敗', data.error || '無法清除事件記錄');
            }
            
        } catch (error) {
            console.error('Clear events error:', error);
            this.showToast('error', '清除失敗', '無法清除事件記錄');
        } finally {
            this.showLoading(false);
        }
    }

    updateSummary() {
        const totalEvents = this.events.length;
        this.totalEvents.textContent = totalEvents;
        this.eventCount.textContent = `總共 ${totalEvents} 個事件`;
        
        if (totalEvents > 0) {
            // 找出最新事件時間
            const sortedEvents = [...this.events].sort((a, b) => new Date(b.time) - new Date(a.time));
            const latestEvent = sortedEvents[0];
            this.latestTime.textContent = latestEvent.time.split(' ')[1]; // 只顯示時間部分
            
            // 計算唯一物件類型數量
            const uniqueObjects = [...new Set(this.events.map(e => e.object))];
            this.objectTypes.textContent = uniqueObjects.length;
        } else {
            this.latestTime.textContent = '--:--';
            this.objectTypes.textContent = '0';
        }
    }

    updateFilters() {
        // 更新物件類型過濾器
        const uniqueObjects = [...new Set(this.events.map(e => e.object))];
        
        // 清除現有選項
        while (this.objectFilter.children.length > 1) {
            this.objectFilter.removeChild(this.objectFilter.lastChild);
        }
        
        // 添加物件類型選項
        uniqueObjects.forEach(obj => {
            const option = document.createElement('option');
            option.value = obj;
            option.textContent = obj;
            this.objectFilter.appendChild(option);
        });
    }

    applyFilters() {
        const selectedObject = this.objectFilter.value;
        
        this.filteredEvents = this.events.filter(event => {
            const objectMatch = selectedObject === 'all' || event.object === selectedObject;
            return objectMatch;
        });
        
        this.displayEvents();
        this.eventCount.textContent = `顯示 ${this.filteredEvents.length} / ${this.events.length} 個事件`;
    }

    displayEvents() {
        this.eventsTableBody.innerHTML = '';
        
        if (this.filteredEvents.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="3" class="no-filtered-events">
                    <i class="fas fa-filter"></i>
                    沒有符合篩選條件的事件
                </td>
            `;
            this.eventsTableBody.appendChild(row);
            return;
        }

        // 按時間排序
        const sortedEvents = [...this.filteredEvents].sort((a, b) => {
            return new Date(b.time) - new Date(a.time);
        });
        
        sortedEvents.forEach(event => {
            const row = document.createElement('tr');
            row.className = 'event-row';
            
            row.innerHTML = `
                <td class="event-time">${event.time}</td>
                <td class="object-type">
                    <span class="object-badge">${event.object}</span>
                </td>
                <td class="floor-info">
                    <span class="floor-badge">${event.floor || '1F'}</span>
                </td>
            `;
            
            this.eventsTableBody.appendChild(row);
        });
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = (seconds % 60).toFixed(1);
        return `${minutes}:${remainingSeconds.padStart(4, '0')}`;
    }

    showLoading(show) {
        this.loadingEvents.style.display = show ? 'block' : 'none';
        if (show) {
            this.noEvents.style.display = 'none';
            this.eventsTable.style.display = 'none';
        }
    }

    showNoEvents() {
        this.loadingEvents.style.display = 'none';
        this.noEvents.style.display = 'block';
        this.eventsTable.style.display = 'none';
    }

    showEventsTable() {
        this.loadingEvents.style.display = 'none';
        this.noEvents.style.display = 'none';
        this.eventsTable.style.display = 'block';
    }

    showToast(type, title, message, duration = 5000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            info: 'fas fa-info-circle',
            warning: 'fas fa-exclamation-triangle'
        };
        
        toast.innerHTML = `
            <i class="${iconMap[type]}"></i>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
        `;
        
        this.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.eventsApp = new EventsApp();
});
