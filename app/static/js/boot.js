/* Start-up. */
(() => {
  'use strict';
  const GS = window.GS;
  GS.route();
  GS.loadStatus();
  GS.refreshUser();
  GS.tour.maybeStart();
})();
