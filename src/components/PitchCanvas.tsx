import React, { useEffect, useRef, useState } from 'react';
import { Ball, Player, TeamConfig, TeamSide, Vector2D } from '../types/football';
import { PITCH } from '../engine/Rules';
import { PolicyActionDistribution } from '../types/telemetry';

interface PitchCanvasProps {
  ball: Ball;
  players: Player[];
  teamLeftConfig: TeamConfig;
  teamRightConfig: TeamConfig;
  controlledPlayerId: string | null;
  cameraMode: 'full' | 'follow';
  onPlayerClick?: (playerId: string) => void;
  onPitchClick?: (pos: Vector2D) => void;
  showRadar?: boolean;
  policyDistribution?: PolicyActionDistribution | null;
  showAttentionVectors?: boolean;
}

export const PitchCanvas: React.FC<PitchCanvasProps> = ({
  ball,
  players,
  teamLeftConfig,
  teamRightConfig,
  controlledPlayerId,
  cameraMode,
  onPlayerClick,
  onPitchClick,
  showRadar = true,
  policyDistribution,
  showAttentionVectors = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 450 });

  // Handle Container Resizing
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        const { width } = entry.contentRect;
        // Maintain 16:9 or 2:1 ratio for football pitch
        const calculatedHeight = Math.max(360, Math.min(650, width * 0.52));
        setDimensions({ width, height: calculatedHeight });
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Main Canvas Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = dimensions;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Padding around pitch in pixels
    const padX = 40;
    const padY = 30;
    const pitchPixelW = width - padX * 2;
    const pitchPixelH = height - padY * 2;

    // Coordinate conversion: GRF coords (-1..1, -0.42..0.42) -> Canvas pixels
    const toCanvasX = (normX: number) => {
      return padX + ((normX - PITCH.minX) / PITCH.width) * pitchPixelW;
    };
    const toCanvasY = (normY: number) => {
      return padY + ((normY - PITCH.minY) / PITCH.height) * pitchPixelH;
    };
    const scaleX = pitchPixelW / PITCH.width;
    const scaleY = pitchPixelH / PITCH.height;

    // 1. Draw Pitch Grass with Stripes
    ctx.fillStyle = '#1b3824'; // rich deep grass
    ctx.fillRect(0, 0, width, height);

    // Mowing stripes (10 vertical stripes)
    const numStripes = 12;
    const stripeWidth = pitchPixelW / numStripes;
    for (let i = 0; i < numStripes; i++) {
      ctx.fillStyle = i % 2 === 0 ? '#22462e' : '#1d3e28';
      ctx.fillRect(padX + i * stripeWidth, padY, stripeWidth, pitchPixelH);
    }

    // 2. Draw Pitch Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 2;

    // Outer Boundary
    ctx.strokeRect(padX, padY, pitchPixelW, pitchPixelH);

    // Halfway Line
    const centerX = toCanvasX(0);
    const centerY = toCanvasY(0);
    ctx.beginPath();
    ctx.moveTo(centerX, padY);
    ctx.lineTo(centerX, padY + pitchPixelH);
    ctx.stroke();

    // Center Circle & Spot
    ctx.beginPath();
    ctx.arc(centerX, centerY, PITCH.centerCircleRadius * scaleY, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 3, 0, Math.PI * 2);
    ctx.fill();

    // Corner Arcs
    const cornerR = PITCH.cornerArcRadius * scaleX;
    // Top-Left
    ctx.beginPath();
    ctx.arc(padX, padY, cornerR, 0, Math.PI * 0.5);
    ctx.stroke();
    // Bottom-Left
    ctx.beginPath();
    ctx.arc(padX, padY + pitchPixelH, cornerR, Math.PI * 1.5, Math.PI * 2);
    ctx.stroke();
    // Top-Right
    ctx.beginPath();
    ctx.arc(padX + pitchPixelW, padY, cornerR, Math.PI * 0.5, Math.PI);
    ctx.stroke();
    // Bottom-Right
    ctx.beginPath();
    ctx.arc(padX + pitchPixelW, padY + pitchPixelH, cornerR, Math.PI, Math.PI * 1.5);
    ctx.stroke();

    // 3. Penalty Boxes & Goal Areas
    // Left Penalty Box
    const penBoxW = PITCH.penaltyBoxLength * scaleX;
    const penBoxH = PITCH.penaltyBoxWidth * scaleY;
    ctx.strokeRect(padX, centerY - penBoxH / 2, penBoxW, penBoxH);

    // Left Goal Area
    const goalBoxW = PITCH.goalBoxLength * scaleX;
    const goalBoxH = PITCH.goalBoxWidth * scaleY;
    ctx.strokeRect(padX, centerY - goalBoxH / 2, goalBoxW, goalBoxH);

    // Left Penalty Spot & Arc
    const leftPenSpotX = toCanvasX(PITCH.minX + PITCH.penaltySpotDist);
    ctx.beginPath();
    ctx.arc(leftPenSpotX, centerY, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(leftPenSpotX, centerY, PITCH.centerCircleRadius * scaleY, -0.65, 0.65);
    ctx.stroke();

    // Right Penalty Box
    ctx.strokeRect(padX + pitchPixelW - penBoxW, centerY - penBoxH / 2, penBoxW, penBoxH);

    // Right Goal Area
    ctx.strokeRect(padX + pitchPixelW - goalBoxW, centerY - goalBoxH / 2, goalBoxW, goalBoxH);

    // Right Penalty Spot & Arc
    const rightPenSpotX = toCanvasX(PITCH.maxX - PITCH.penaltySpotDist);
    ctx.beginPath();
    ctx.arc(rightPenSpotX, centerY, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(rightPenSpotX, centerY, PITCH.centerCircleRadius * scaleY, Math.PI - 0.65, Math.PI + 0.65);
    ctx.stroke();

    // 4. Goal Nets & Posts
    const goalH = PITCH.goalWidth * scaleY;
    const goalDepth = 18;

    // Left Goal Net
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.fillRect(padX - goalDepth, centerY - goalH / 2, goalDepth, goalH);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(padX - goalDepth, centerY - goalH / 2, goalDepth, goalH);

    // Right Goal Net
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.fillRect(padX + pitchPixelW, centerY - goalH / 2, goalDepth, goalH);
    ctx.strokeRect(padX + pitchPixelW, centerY - goalH / 2, goalDepth, goalH);

    // 5. Draw Ball Trail
    if (ball.trail && ball.trail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(ball.trail[0].x), toCanvasY(ball.trail[0].y));
      for (let i = 1; i < ball.trail.length; i++) {
        const opacity = (i / ball.trail.length) * 0.4;
        ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.lineTo(toCanvasX(ball.trail[i].x), toCanvasY(ball.trail[i].y));
      }
      ctx.stroke();
    }

    // 5.5 Draw Neural Policy Spatial Attention & Action Cones
    if (showAttentionVectors && policyDistribution) {
      const activePlayer = players.find((p) => p.id === policyDistribution.playerId);
      if (activePlayer) {
        const apx = toCanvasX(activePlayer.position.x);
        const apy = toCanvasY(activePlayer.position.y);

        // 5.5.1 Passing attention vector to target receiver
        if (policyDistribution.attention?.targetPos) {
          const tpx = toCanvasX(policyDistribution.attention.targetPos.x);
          const tpy = toCanvasY(policyDistribution.attention.targetPos.y);

          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([5, 5]);
          ctx.moveTo(apx, apy);
          ctx.lineTo(tpx, tpy);
          ctx.strokeStyle = 'rgba(6, 182, 212, 0.85)';
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.setLineDash([]);

          // Glowing receiver target marker
          ctx.fillStyle = 'rgba(6, 182, 212, 0.3)';
          ctx.beginPath();
          ctx.arc(tpx, tpy, 16, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#06b6d4';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // Lane clearance label badge
          const midX = (apx + tpx) / 2;
          const midY = (apy + tpy) / 2;
          const clearanceText = `Pass Lane: ${policyDistribution.attention.passClearanceProb}%`;
          ctx.font = 'bold 9px "Plus Jakarta Sans", sans-serif';
          const textW = ctx.measureText(clearanceText).width;

          ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
          ctx.fillRect(midX - textW / 2 - 4, midY - 7, textW + 8, 14);
          ctx.strokeStyle = '#06b6d4';
          ctx.lineWidth = 1;
          ctx.strokeRect(midX - textW / 2 - 4, midY - 7, textW + 8, 14);

          ctx.fillStyle = '#22d3ee';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(clearanceText, midX, midY);
          ctx.restore();
        }

        // 5.5.2 Shot corridor to opponent goal
        if (policyDistribution.attention?.shotAngleClearance && activePlayer.position.x > 0.1) {
          const goalTopX = toCanvasX(1.0);
          const goalTopY = toCanvasY(-0.07);
          const goalBotX = toCanvasX(1.0);
          const goalBotY = toCanvasY(0.07);

          ctx.save();
          ctx.beginPath();
          ctx.moveTo(apx, apy);
          ctx.lineTo(goalTopX, goalTopY);
          ctx.lineTo(goalBotX, goalBotY);
          ctx.closePath();
          ctx.fillStyle = 'rgba(245, 158, 11, 0.1)';
          ctx.fill();

          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.restore();
        }

        // 5.5.3 Centralized Critic Value V(s) Ring around player
        const val = policyDistribution.valueEstimate;
        const ringColor = val > 0.3 ? '#10b981' : val > -0.1 ? '#f59e0b' : '#ef4444';
        ctx.save();
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(apx, apy, 20, 0, Math.PI * 2);
        ctx.stroke();

        // Value text tag
        const vText = `V(s): ${val >= 0 ? `+${val}` : val}`;
        ctx.font = 'bold 9px "Plus Jakarta Sans", sans-serif';
        const vTextW = ctx.measureText(vText).width;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
        ctx.fillRect(apx - vTextW / 2 - 3, apy + 20, vTextW + 6, 12);
        ctx.fillStyle = ringColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(vText, apx, apy + 26);
        ctx.restore();
      }
    }

    // 6. Draw Players
    const playerRadiusPx = Math.max(9, PITCH.playerRadius * scaleY);

    players.forEach((player) => {
      const px = toCanvasX(player.position.x);
      const py = toCanvasY(player.position.y);
      const isControlled = player.id === controlledPlayerId;
      const isLeft = player.team === 'left';
      const teamConfig = isLeft ? teamLeftConfig : teamRightConfig;

      // Player Shadow
      ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
      ctx.beginPath();
      ctx.ellipse(px, py + 3, playerRadiusPx * 0.9, playerRadiusPx * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();

      // Heading Indicator Cone / Arrow
      const headingX = px + Math.cos(player.heading) * (playerRadiusPx + 5);
      const headingY = py + Math.sin(player.heading) * (playerRadiusPx + 5);
      ctx.strokeStyle = isControlled ? '#fbbf24' : 'rgba(255, 255, 255, 0.6)';
      ctx.lineWidth = isControlled ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(headingX, headingY);
      ctx.stroke();

      // Selection Marker Ring if user-controlled
      if (isControlled) {
        // Glowing inverted indicator triangle above player
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.moveTo(px, py - playerRadiusPx - 10);
        ctx.lineTo(px - 5, py - playerRadiusPx - 16);
        ctx.lineTo(px + 5, py - playerRadiusPx - 16);
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, playerRadiusPx + 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Slide tackle animation effect
      if (player.isTackling) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(px, py, playerRadiusPx + 6, player.heading - 0.8, player.heading + 0.8);
        ctx.stroke();
      }

      // Main Player Circle Body
      ctx.fillStyle = player.isGoalkeeper ? '#eab308' : teamConfig.color;
      ctx.beginPath();
      ctx.arc(px, py, playerRadiusPx, 0, Math.PI * 2);
      ctx.fill();

      // Inner Accent Border
      ctx.strokeStyle = teamConfig.accentColor || '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Player Number & Role Label
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${Math.max(8, playerRadiusPx * 0.9)}px 'Plus Jakarta Sans', sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${player.number}`, px, py);

      // Stamina Indicator Arc
      if (player.stamina < 95) {
        const staminaAngle = (player.stamina / 100) * Math.PI * 2;
        ctx.strokeStyle = player.stamina > 40 ? '#22c55e' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, playerRadiusPx + 2, -Math.PI / 2, -Math.PI / 2 + staminaAngle);
        ctx.stroke();
      }
    });

    // 7. Draw Ball
    const ballX = toCanvasX(ball.position.x);
    const ballY = toCanvasY(ball.position.y);
    const ballZ = ball.position.z * scaleY * 12; // Z-axis lift height
    const ballRadiusPx = Math.max(5, PITCH.ballRadius * scaleY);

    // Ball Ground Shadow (displaced by Z)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.beginPath();
    ctx.ellipse(ballX, ballY, ballRadiusPx * 1.1, ballRadiusPx * 0.6, 0, 0, Math.PI * 2);
    ctx.fill();

    // Actual Elevated Ball
    const renderBallY = ballY - ballZ;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(ballX, renderBallY, ballRadiusPx, 0, Math.PI * 2);
    ctx.fill();

    // Ball Pentagons / Details
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(ballX, renderBallY, ballRadiusPx * 0.45, 0, Math.PI * 2);
    ctx.fill();

    // 8. Mini-Radar (Optional Overlay in Bottom-Right)
    if (showRadar) {
      const radarW = 140;
      const radarH = 65;
      const radarX = width - radarW - 15;
      const radarY = height - radarH - 15;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.fillRect(radarX, radarY, radarW, radarH);
      ctx.strokeRect(radarX, radarY, radarW, radarH);

      // Radar center line
      ctx.beginPath();
      ctx.moveTo(radarX + radarW / 2, radarY);
      ctx.lineTo(radarX + radarW / 2, radarY + radarH);
      ctx.stroke();

      // Radar players
      players.forEach((p) => {
        const rx = radarX + ((p.position.x - PITCH.minX) / PITCH.width) * radarW;
        const ry = radarY + ((p.position.y - PITCH.minY) / PITCH.height) * radarH;
        ctx.fillStyle = p.team === 'left' ? '#3b82f6' : '#ef4444';
        ctx.beginPath();
        ctx.arc(rx, ry, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Radar ball
      const rbx = radarX + ((ball.position.x - PITCH.minX) / PITCH.width) * radarW;
      const rby = radarY + ((ball.position.y - PITCH.minY) / PITCH.height) * radarH;
      ctx.fillStyle = '#facc15';
      ctx.beginPath();
      ctx.arc(rbx, rby, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [
    dimensions,
    ball,
    players,
    teamLeftConfig,
    teamRightConfig,
    controlledPlayerId,
    showRadar,
    policyDistribution,
    showAttentionVectors,
  ]);

  // Click on Canvas handler
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const padX = 40;
    const padY = 30;
    const pitchPixelW = dimensions.width - padX * 2;
    const pitchPixelH = dimensions.height - padY * 2;

    const normX = PITCH.minX + ((clickX - padX) / pitchPixelW) * PITCH.width;
    const normY = PITCH.minY + ((clickY - padY) / pitchPixelH) * PITCH.height;

    // Check if clicked directly on a player
    for (const p of players) {
      const px = padX + ((p.position.x - PITCH.minX) / PITCH.width) * pitchPixelW;
      const py = padY + ((p.position.y - PITCH.minY) / PITCH.height) * pitchPixelH;
      const dist = Math.hypot(clickX - px, clickY - py);
      if (dist < 18) {
        onPlayerClick?.(p.id);
        return;
      }
    }

    onPitchClick?.({ x: normX, y: normY });
  };

  return (
    <div
      ref={containerRef}
      id="pitch-container"
      className="relative w-full rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl shadow-emerald-950/20"
    >
      <canvas
        ref={canvasRef}
        id="pitch-canvas"
        onClick={handleCanvasClick}
        style={{ width: dimensions.width, height: dimensions.height }}
        className="block cursor-crosshair"
      />
    </div>
  );
};
