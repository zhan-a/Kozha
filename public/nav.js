/* Shared site-nav behavior. Every public page loads this once to wire
 * up the mobile hamburger dropdown on .kz-header. Deliberately tiny and
 * dependency-free so it can run before any other script. */
(function () {
  'use strict';

  var header = document.querySelector('.kz-header');
  if (!header) return;

  var hamburger = header.querySelector('.kz-header__hamburger');
  var nav = header.querySelector('.kz-header__nav');
  if (!hamburger || !nav) return;

  function setOpen(open) {
    header.classList.toggle('is-open', open);
    hamburger.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  hamburger.addEventListener('click', function () {
    setOpen(!header.classList.contains('is-open'));
  });

  nav.addEventListener('click', function (e) {
    if (e.target && e.target.tagName === 'A') setOpen(false);
  });

  // Close the dropdown if the viewport grows past the mobile breakpoint.
  var mq = window.matchMedia('(min-width: 768px)');
  var closeOnDesktop = function (e) { if (e.matches) setOpen(false); };
  if (typeof mq.addEventListener === 'function') {
    mq.addEventListener('change', closeOnDesktop);
  } else if (typeof mq.addListener === 'function') {
    mq.addListener(closeOnDesktop);
  }

  // Close on Escape when open.
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && header.classList.contains('is-open')) {
      setOpen(false);
      hamburger.focus();
    }
  });
})();
